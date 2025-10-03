from typing import Optional

import os
import requests

try:
    from transformers import MarianMTModel, MarianTokenizer
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False


class Translator:
    """
    Translator for English ↔ Arabic using MarianMT when available.
    Falls back to LibreTranslate public API if local models are unavailable.
    Identity translations (src == tgt) return text unchanged.
    """

    def __init__(self, source_lang: str = "en", target_lang: str = "ar"):
        self.source_lang = source_lang.lower()
        self.target_lang = target_lang.lower()
        self._model_name = None
        self._tokenizer = None
        self._model = None

        # LibreTranslate endpoint (can be overridden via env)
        self._libre_endpoint = os.getenv("LIBRETRANSLATE_URL", "https://libretranslate.com/translate")

        if self.source_lang == self.target_lang:
            # No model needed for identity translation
            return

        if _TRANSFORMERS_AVAILABLE:
            self._model_name = self._select_model_name(self.source_lang, self.target_lang)
            try:
                self._tokenizer = MarianTokenizer.from_pretrained(self._model_name)
                self._model = MarianMTModel.from_pretrained(self._model_name)
            except Exception:
                self._tokenizer = None
                self._model = None

    @staticmethod
    def _select_model_name(src: str, tgt: str) -> str:
        pair = (src, tgt)
        if pair == ("en", "ar"):
            return "Helsinki-NLP/opus-mt-en-ar"
        if pair == ("ar", "en"):
            return "Helsinki-NLP/opus-mt-ar-en"
        # Fallback to English → Arabic
        return "Helsinki-NLP/opus-mt-en-ar"

    def _translate_remote(self, text: str, source_lang: str, target_lang: str) -> str:
        if source_lang == target_lang:
            return text
        try:
            r = requests.post(
                self._libre_endpoint,
                data={"q": text, "source": source_lang, "target": target_lang, "format": "text"},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=20,
            )
            if r.ok:
                data = r.json()
                return data.get("translatedText", text)
        except Exception:
            pass
        return text

    def translate(self, text: str, source_lang: Optional[str] = None, target_lang: Optional[str] = None) -> str:
        src = (source_lang or self.source_lang).lower()
        tgt = (target_lang or self.target_lang).lower()

        if src == tgt:
            return text

        if self._model and self._tokenizer:
            if (src, tgt) != (self.source_lang, self.target_lang):
                # reinitialize model if language pair changed
                self.source_lang, self.target_lang = src, tgt
                self._model_name = self._select_model_name(src, tgt)
                try:
                    self._tokenizer = MarianTokenizer.from_pretrained(self._model_name)
                    self._model = MarianMTModel.from_pretrained(self._model_name)
                except Exception:
                    self._tokenizer = None
                    self._model = None

        if self._model and self._tokenizer:
            batch = self._tokenizer([text], return_tensors="pt", padding=True)
            gen = self._model.generate(**batch, max_new_tokens=256)
            out = self._tokenizer.batch_decode(gen, skip_special_tokens=True)
            return out[0]

        # Fallback to remote API
        return self._translate_remote(text, src, tgt)
