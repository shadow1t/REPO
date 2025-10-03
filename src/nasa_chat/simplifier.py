from typing import Literal

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


class Simplifier:
    """
    Rewrite technical text into simpler English for non-experts.
    Uses FLAN-T5 for instruction-following rewriting.
    """

    def __init__(self, model_name: str = "google/flan-t5-base"):
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._pipe = pipeline("text2text-generation", model=mdl, tokenizer=tok)

    def simplify(
        self,
        text: str,
        audience: Literal["child", "general"] = "child",
    ) -> str:
        style = "for a 10-year-old child" if audience == "child" else "for a general audience"
        prompt = (
            f"Rewrite the following explanation in English {style}. "
            f"Use bullet points with short sentences, avoid jargon, define key terms briefly, "
            f"keep facts correct, and add one simple analogy.\n\n{text}"
        )
        res = self._pipe(prompt, max_new_tokens=256, do_sample=False)
        return res[0]["generated_text"]