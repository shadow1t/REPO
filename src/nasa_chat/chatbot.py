from typing import Any, Dict, List, Optional

from .nasa_api import NASAAPI
from .simplifier import Simplifier
from .translator import Translator
from .captioner import ImageCaptioner


class ChatBot:
    """
    NASA-aware chat assistant that can:
    - Answer text questions using NASA Image and Video Library search
    - Describe uploaded images (optional) using BLIP captioner
    - Translate answers to Arabic or English
    - Simplify explanations for non-experts or kids
    """

    def __init__(
        self,
        language: str = "ar",
        use_vision: bool = True,
        api_key: Optional[str] = None,
        simplifier_model: str = "google/flan-t5-base",
        audience: str = "child",
    ):
        self.language = language.lower()
        self.nasa = NASAAPI(api_key=api_key)
        self.simplifier = Simplifier(model_name=simplifier_model)
        self.audience = audience if audience in {"child", "general"} else "child"
        self.translator = Translator(source_lang="en", target_lang=self.language)
        self.captioner = ImageCaptioner() if use_vision else None
        self.history: List[Dict[str, Any]] = []

    def status(self) -> Dict[str, bool]:
        return {
            "has_blip": bool(self.captioner and getattr(self.captioner, "model", None)),
            "has_flan": bool(getattr(self.simplifier, "_pipe", None)),
            "has_marian": bool(getattr(self.translator, "_model", None)),
        }

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Handle a text-only query. Strategy:
        - If target language is Arabic, translate query to English for NASA search
        - Search NASA Image Library with the (English) query
        - Aggregate top results descriptions as technical context
        - Simplify in English, then translate if target language is Arabic
        """
        english_query = (
            self.translator.translate(query, source_lang=self.language, target_lang="en")
            if self.language != "en"
            else query
        ).strip()

        results = self.nasa.search_images(english_query)
        if not results:
            # Second-pass expansions to improve recall
            expansions = []
            ql = english_query.lower()
            if "earth" not in ql:
                expansions.append(f"{english_query} earth")
            if "ocean" not in ql and "sea" not in ql:
                expansions.extend(["ocean", "sea", "earth oceans", "ocean currents"])
            for ex in expansions:
                try:
                    results = self.nasa.search_images(ex)
                    if results:
                        english_query = ex
                        break
                except Exception:
                    continue

        if not results:
            technical = f"No NASA image results found for: {query}"
            simple_en = self.simplifier.simplify(
                "Explain Earth's oceans from a NASA perspective: satellites observe sea surface temperature, sea level, "
                "currents, color (chlorophyll), and storms. Why oceans matter and how space helps us monitor them.",
                audience=self.audience,
            )
            simple = (
                self.translator.translate(simple_en, source_lang="en", target_lang="ar")
                if self.language == "ar"
                else simple_en
            )
            resp = {"technical": technical, "simple": simple, "sources": []}
            self._remember(query, resp)
            return resp

        # Aggregate top descriptions for richer context
        k = min(5, len(results))
        selected = results[:k]
        descriptions: List[str] = []
        for it in selected:
            d = it.get("description") or it.get("title") or ""
            if d:
                descriptions.append(d)
        technical_context = (
            f"NASA results overview for query '{english_query}':\n\n" + "\n\n".join(descriptions)
            if descriptions
            else (selected[0].get("description") or selected[0].get("title") or "NASA result")
        )

        simple_en = self.simplifier.simplify(technical_context, audience=self.audience)
        simple = (
            self.translator.translate(simple_en, source_lang="en", target_lang="ar")
            if self.language == "ar"
            else simple_en
        )
        formatted_sources = [
            {"title": it.get("title"), "preview_url": it.get("preview_url"), "nasa_id": it.get("nasa_id")}
            for it in selected
        ]
        resp = {"technical": technical_context, "simple": simple, "sources": formatted_sources}
        self._remember(query, resp)
        return resp

    def describe_image(self, image_path: str) -> Dict[str, Any]:
        """
        Handle an image by captioning and then searching NASA with the caption keywords.
        Uses intelligent query expansions to improve recall (e.g., sun → SDO, SOHO, solar flare).
        Aggregates multiple NASA descriptions for richer context.
        """
        if not self.captioner:
            technical = "Vision module is disabled."
            simple_en = self.simplifier.simplify(technical, audience=self.audience)
            simple = (
                self.translator.translate(simple_en, source_lang="en", target_lang="ar")
                if self.language == "ar"
                else simple_en
            )
            resp = {"technical": technical, "simple": simple, "sources": []}
            self._remember(f"[image]{image_path}", resp)
            return resp

        caption = self.captioner.caption(image_path).strip()
        cq = caption.lower()

        # Build query expansions
        queries: List[str] = [caption]
        # Correct common mis-caption like "hub" → "Hubble"
        if " hub " in f" {cq} ":
            queries.append("hubble")
            queries.append("hubble telescope")
        # Sun/solar related expansions
        if any(k in cq for k in ["sun", "solar", "corona", "sunspot", "flare"]):
            queries.extend([
                "sun", "solar", "SDO", "SOHO", "solar dynamics observatory",
                "parker solar probe", "solar flare", "sunspots", "coronal mass ejection"
            ])
        # Earth/general expansions
        if len(queries) == 1:
            if any(k in cq for k in ["cloud", "atmosphere"]):
                queries.extend(["earth clouds", "atmosphere earth from space", "EPIC Earth"])
            elif any(k in cq for k in ["ocean", "sea"]):
                queries.extend(["earth oceans", "ocean currents satellite", "sea surface temperature"])
            elif any(k in cq for k in ["forest", "vegetation", "green"]):
                queries.extend(["earth vegetation", "NDVI", "forest earth from space"])
            else:
                queries.extend(["earth from space", "satellite earth image"])

        # Try searches until we get results
        results: List[Dict[str, Any]] = []
        for q in queries:
            try:
                r = self.nasa.search_images(q)
            except Exception:
                r = []
            if r:
                results = r
                break

        if not results:
            technical = f"Image caption: {caption}. No related NASA results found after expansions."
            # Provide a helpful fallback explanation tailored to caption theme
            if any(k in cq for k in ["sun", "solar"]):
                fallback = (
                    "NASA studies the Sun using space missions like SDO and SOHO. "
                    "They observe sunspots, solar flares, and the corona to understand space weather, "
                    "which can affect satellites and power grids on Earth."
                )
            else:
                fallback = (
                    "Satellites observe Earth to measure clouds, oceans, forests, and atmosphere. "
                    "These observations help scientists track weather, climate, and environmental changes."
                )
            simple_en = self.simplifier.simplify(f"{technical}\n\n{fallback}", audience=self.audience)
            simple = (
                self.translator.translate(simple_en, source_lang="en", target_lang="ar")
                if self.language == "ar"
                else simple_en
            )
            resp = {"technical": technical, "simple": simple, "sources": []}
            self._remember(f"[image]{image_path}", resp)
            return resp

        # Aggregate multiple results for richer context
        k = min(5, len(results))
        selected = results[:k]
        descriptions: List[str] = []
        for it in selected:
            d = it.get("description") or it.get("title") or ""
            if d:
                descriptions.append(d)
        technical_context = (
            f"Image caption: {caption}\n\nNASA context from top results:\n\n" + "\n\n".join(descriptions)
            if descriptions
            else f"Image caption: {caption}\n\nNASA context: " + (selected[0].get("title") or "NASA result")
        )

        simple_en = self.simplifier.simplify(technical_context, audience=self.audience)
        simple = (
            self.translator.translate(simple_en, source_lang="en", target_lang="ar")
            if self.language == "ar"
            else simple_en
        )
        formatted_sources = [
            {"title": it.get("title"), "preview_url": it.get("preview_url"), "nasa_id": it.get("nasa_id")}
            for it in selected
        ]
        resp = {"technical": technical_context, "simple": simple, "sources": formatted_sources}
        self._remember(f"[image]{image_path}", resp)
        return resp

    def _remember(self, user_input: str, response: Dict[str, Any]) -> None:
        self.history.append({"input": user_input, "response": response})

    def reset(self) -> None:
        self.history.clear()
