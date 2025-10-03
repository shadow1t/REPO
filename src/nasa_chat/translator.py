from typing import Optional

from transformers import MarianMTModel, MarianTokenizer


class Translator:
    """
    Lightweight translator for English ↔ Arabic using MarianMT.
    """

    def __init__(self, source_lang: str = "en", target_lang: str = "ar"):
        self.source_lang = source_lang.lower()
        self.target_lang = target_lang.lower()
        self._model_name = self._select_model_name(self.source_lang, self.target_lang)
        self._tokenizer = MarianTokenizer.from_pretrained(self._model_name)
        self._model = MarianMTModel.from_pretrained(self._model_name)

    @staticmethod
    def _select_model_name(src: str, tgt: str) -> str:
        pair = (src, tgt)
        if pair == ("en", "ar"):
            return "Helsinki-NLP/opus-mt-en-ar"
        if pair == ("ar", "en"):
            return "Helsinki-NLP/opus-mt-ar-en"
        # Fallback to English → Arabic
        return "Helsinki-NLP/opus-mt-en-ar"

    def translate(self, text: str, source_lang: Optional[str] = None, target_lang: Optional[str] = None) -> str:
        src = (source_lang or self.source_lang).lower()
        tgt = (target_lang or self.target_lang).lower()
        if (src, tgt) != (self.source_lang, self.target_lang):
            # reinitialize model if language pair changed
            self.source_lang, self.target_lang = src, tgt
            self._model_name = self._select_model_name(src, tgt)
            self._tokenizer = MarianTokenizer.from_pretrained(self._model_name)
            self._model = MarianMTModel.from_pretrained(self._model_name)

        batch = self._tokenizer([text], return_tensors="pt", padding=True)
        gen = self._model.generate(**batch, max_new_tokens=256)
        out = self._tokenizer.batch_decode(gen, skip_special_tokens=True)
        return out[0]