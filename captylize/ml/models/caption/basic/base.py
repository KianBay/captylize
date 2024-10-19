from PIL import Image
from captylize.ml.models.img_to_text_model import Img2TextModel


class BasicCaptionModel(Img2TextModel[str]):
    def _predict(self, image: Image.Image) -> str:
        raise NotImplementedError
