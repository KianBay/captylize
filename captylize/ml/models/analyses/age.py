from typing import Literal
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from captylize.ml.models.ml_model import Img2TextModel


class ViTAgeClassifier(Img2TextModel):
    def __init__(
        self,
        cache_dir: str,
        device: Literal["cpu", "cuda", "mps"],
        use_safetensors: bool = True,
    ):
        super().__init__(cache_dir, device, use_safetensors)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.transforms is not None

    def load(self):
        self.model = ViTForImageClassification.from_pretrained(
            "nateraw/vit-age-classifier",
            cache_dir=self.cache_dir,
            use_safetensors=self.use_safetensors,
        ).to(self.device)

        self.transforms = ViTFeatureExtractor.from_pretrained(
            "nateraw/vit-age-classifier", cache_dir=self.cache_dir
        )

    def unload(self):
        self.model = None
        self.transforms = None

    def predict(self, image: Image.Image):
        inputs = self.transforms(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = outputs.logits.softmax(1)

        result = {
            self.model.config.id2label[i]: prob.item()
            for i, prob in enumerate(probabilities)
        }

        return result
