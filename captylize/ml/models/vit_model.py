from typing import Literal
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

from captylize.ml.models.ml_model import Img2TextModel
from captylize.logger import get_logger

logger = get_logger(__name__)


class ViTImg2TextModel(Img2TextModel[dict[str, float]]):
    def __init__(
        self,
        model_location: str,
        cache_dir: str,
        device: Literal["cpu", "cuda", "mps"],
        use_safetensors: bool = True,
    ):
        super().__init__(model_location, cache_dir, device, use_safetensors)

    def _load(self) -> None:
        logger.info(f"Loading model {self.__class__.__name__}")
        self.model = ViTForImageClassification.from_pretrained(
            self.model_location,
            cache_dir=self.cache_dir,
            use_safetensors=self.use_safetensors,
        ).to(self.device)

        self.processor = ViTImageProcessor.from_pretrained(
            self.model_location, cache_dir=self.cache_dir
        )

    def _unload(self) -> None:
        self.model = None
        self.processor = None

    def _predict(self, image: Image.Image) -> dict[str, float]:
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = outputs.logits.softmax(1)
        probabilities = probabilities.squeeze().cpu().tolist()

        result = {
            self.model.config.id2label[i]: prob for i, prob in enumerate(probabilities)
        }

        return result
