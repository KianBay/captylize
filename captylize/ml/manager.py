import torch
import os


from typing import Literal

from captylize.ml.models.ml_model import Img2TextModel
from captylize.ml.models.analyses.age import ViTAgeClassifier
from captylize.ml.models.analyses.emotion import VitEmotionClassifier


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

        self.emotion_models: dict[str, Img2TextModel] = {
            "VitEmotionClassifier": VitEmotionClassifier(
                cache_dir=self.cache_dir, device=self.device
            )
        }
        self.default_emotion_model: Literal["VitEmotionClassifier"] = (
            "VitEmotionClassifier"
        )

    def load_models(self):
        for model in self.age_models.values():
            model.load()
        for model in self.emotion_models.values():
            model.load()

    def unload_models(self):
        for model in self.age_models.values():
            model.unload()

    def get_age_model(self, model_name: str = None) -> Img2TextModel:
        if model_name and model_name in self.age_models:
            return self.age_models[model_name]
        return self.age_models[self.default_age_model]

    def get_emotion_model(self, model_name: str = None) -> Img2TextModel:
        if model_name and model_name in self.emotion_models:
            return self.emotion_models[model_name]
        return self.emotion_models[self.default_emotion_model]


DEVICE = get_device()
model_manager = ModelManager(cache_dir="./model_cache", device=DEVICE)
