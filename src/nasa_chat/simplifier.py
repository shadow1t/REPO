from typing import Literal

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False


class Simplifier:
    """
    Simplify technical text into clearer English.
    Uses FLAN-T5 when available, otherwise rule-based fallback.
    """

    def __init__(self, model_name: str = "google/flan-t5-base"):
        self._pipe = None
        if _TRANSFORMERS_AVAILABLE:
            try:
                tok = AutoTokenizer.from_pretrained(model_name)
                mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self._pipe = pipeline("text2text-generation", model=mdl, tokenizer=tok)
            except Exception:
                self._pipe = None

    def _fallback(self, text: str, audience: str) -> str:
        glossary = {
            "satellite": "a machine in space that watches or communicates",
            "orbit": "the path an object takes around a planet",
            "spectral": "about colors/wavelengths of light",
            "radiance": "brightness of light measured",
            "precipitation": "rain or snow",
            "albedo": "how much light a surface reflects",
            "currents": "moving water in the ocean",
            "chlorophyll": "green stuff in tiny plants (phytoplankton) in the ocean",
        }
        lines = []
        lines.append("- We explain this topic in simple words.")
        for k, v in glossary.items():
            if k in text.lower():
                lines.append(f"- {k}: {v}")
        # Short sentences
        parts = [p.strip() for p in text.split("\n") if p.strip()]
        for p in parts:
            if len(p) > 160:
                lines.append(f"- {p[:160]}...")
            else:
                lines.append(f"- {p}")
        if audience == "child":
            lines.append("- Imagine it like a big camera in space watching Earth to help us understand it.")
        else:
            lines.append("- Analogy: spaceborne cameras monitor Earth much like weather stations monitor a city.")
        return "\n".join(lines)

    def simplify(
        self,
        text: str,
        audience: Literal["child", "general"] = "child",
    ) -> str:
        style = "for a 10-year-old child" if audience == "child" else "for a general audience"
        if self._pipe:
            prompt = (
                f"Rewrite the following explanation in English {style}. "
                f"Use bullet points with short sentences, avoid jargon, define key terms briefly, "
                f"keep facts correct, and add one simple analogy.\n\n{text}"
            )
            res = self._pipe(prompt, max_new_tokens=256, do_sample=False)
            return res[0]["generated_text"]
        return self._fallback(text, audience)
