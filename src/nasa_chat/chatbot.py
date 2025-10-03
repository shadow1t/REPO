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

        caption = self.captioner.caption(image_path)
        results = self.nasa.search_images(caption)
        if not results:
            technical = f"Caption: {caption}. No related NASA results found."
            simple_en = self.simplifier.simplify(technical, audience=self.audience)
            simple = (
                self.translator.translate(simple_en, source_lang="en", target_lang="ar")
                if self.language == "ar"
                else simple_en
            )
            resp = {"technical": technical, "simple": simple, "sources": []}
            self._remember(f"[image]{image_path}", resp)
            return resp

        top = results[0]
        technical_desc = top.get("description") or top.get("title") or "NASA result"
        combined = f"Image caption: {caption}\n\nNASA context: {technical_desc}"
        simple_en = self.simplifier.simplify(combined, audience=self.audience)
        simple = (
            self.translator.translate(simple_en, source_lang="en", target_lang="ar")
            if self.language == "ar"
            else simple_en
        )
        formatted_sources = [
            {"title": top.get("title"), "preview_url": top.get("preview_url"), "nasa_id": top.get("nasa_id")}
        ]
        resp = {"technical": combined, "simple": simple, "sources": formatted_sources}
        self._remember(f"[image]{image_path}", resp)
        return resp

    def _remember(self, user_input: str, response: Dict[str, Any]) -> None:
        self.history.append({"input": user_input, "response": response})

    def reset(self) -> None:
        self.history.clear()
