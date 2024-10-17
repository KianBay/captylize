import torch
import os


from typing import Literal

from captylize.ml.models.ml_model import Img2TextModel
from captylize.ml.models.analyses.age import ViTAgeClassifier


def get_device() -> Literal["cpu", "cuda", "mps"]:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


class ModelManager:
    def __init__(
        self,
        cache_dir: str = "./model_cache",
        device: Literal["cpu", "cuda", "mps"] = "cpu",
    ):
        self.cache_dir = cache_dir
        self.device = device

        self.age_models: dict[str, Img2TextModel] = {
            "ViTAgeClassifier": ViTAgeClassifier(
                cache_dir=self.cache_dir, device=self.device
            )
        }
        self.default_age_model: Literal["ViTAgeClassifier"] = "ViTAgeClassifier"

    def load_models(self):
        for model in self.age_models.values():
            model.load()

    def unload_models(self):
        for model in self.age_models.values():
            model.unload()
