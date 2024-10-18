from typing import Literal
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from captylize.logger import get_logger
from captylize.ml.models.ml_model import Img2TextModel

logger = get_logger(__name__)


class ViTAgeClassifier(Img2TextModel):
    def __init__(
        self,
        cache_dir: str,
        device: Literal["cpu", "cuda", "mps"],
        use_safetensors: bool = True,
    ):
        super().__init__(cache_dir, device, use_safetensors)

    def _load(self):
        logger.info(f"Loading model {self.__class__.__name__}")
        self.model = ViTForImageClassification.from_pretrained(
            "nateraw/vit-age-classifier",
            cache_dir=self.cache_dir,
            use_safetensors=self.use_safetensors,
        ).to(self.device)

        self.processor = ViTImageProcessor.from_pretrained(
            "nateraw/vit-age-classifier", cache_dir=self.cache_dir
        )

    def _unload(self):
        logger.info(f"Unloading model {self.__class__.__name__}")
        self.model = None
        self.processor = None

    def _predict(self, image: Image.Image):
        if not self.is_loaded:
            logger.debug(
                "Predict was called, but model was not loaded. Loading model..."
            )
            self.load()
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = outputs.logits.softmax(1)
        probabilities = probabilities.squeeze().cpu().tolist()

        result = {
            self.model.config.id2label[i]: prob for i, prob in enumerate(probabilities)
        }

        return result
