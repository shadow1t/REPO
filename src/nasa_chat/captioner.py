from typing import Optional

from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


class ImageCaptioner:
    """
    CPU-friendly image captioning using BLIP base.
    """

    def __init__(self):
        model_name = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

    def caption(self, image_path: str, prompt: Optional[str] = None) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=prompt or "", return_tensors="pt")
        out = self.model.generate(**inputs, max_new_tokens=64)
        caption = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        return caption.strip()