from typing import Optional
from PIL import Image
from captylize.ml.models.img_to_text_model import Img2TextModel


class CaptionModel(Img2TextModel[str]):
    def _predict(self, image: Image.Image) -> str:
        raise NotImplementedError


class PromptableCaptionModel(Img2TextModel[str]):
    def _predict(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        raise NotImplementedError
