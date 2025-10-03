from typing import Optional, Tuple

from PIL import Image, ImageStat

try:
    from transformers import BlipForConditionalGeneration, BlipProcessor
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False


class ImageCaptioner:
    """
    Image captioning using BLIP if available.
    Falls back to a lightweight heuristic captioner on CPU-only environments.
    """

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        self.model = None
        self.processor = None
        if _TRANSFORMERS_AVAILABLE:
            try:
                self.processor = BlipProcessor.from_pretrained(model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            except Exception:
                # Fallback if model cannot be loaded (e.g., torch unavailable)
                self.model = None
                self.processor = None

    def _heuristic_keywords(self, image: Image.Image) -> Tuple[str, str]:
        # Compute average color to guess simple keywords
        image_small = image.copy()
        image_small.thumbnail((64, 64))
        stat = ImageStat.Stat(image_small)
        r, g, b = stat.mean[:3]
        kws = []
        if b > r and b > g and b > 90:
            kws.extend(["ocean", "water", "earth"])
        if r > g and r > b and r > 110:
            kws.extend(["mars", "red planet"])
        if g > r and g > b and g > 110:
            kws.extend(["forest", "vegetation", "earth"])
        if r > 150 and g > 150 and b > 150:
            kws.extend(["clouds", "atmosphere", "earth"])
        if not kws:
            kws.extend(["earth", "satellite"])
        desc = "A photo likely showing " + ", ".join(kws[:3])
        return desc, " ".join(kws)

    def caption(self, image_path: str, prompt: Optional[str] = None) -> str:
        image = Image.open(image_path).convert("RGB")
        if self.model and self.processor:
            inputs = self.processor(images=image, text=prompt or "", return_tensors="pt")
            out = self.model.generate(**inputs, max_new_tokens=64)
            caption = self.processor.batch_decode(out, skip_special_tokens=True)[0]
            return caption.strip()
        # Fallback
        desc, _ = self._heuristic_keywords(image)
        return desc
