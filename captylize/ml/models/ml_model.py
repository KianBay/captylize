from abc import ABC, abstractmethod
from typing import Literal
from PIL import Image


class Img2TextModel(ABC):
    def __init__(self, cache_dir: str, device: Literal["cpu", "cuda", "mps"]):
        self.cache_dir = cache_dir
        self.device = device

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def unload(self):
        pass

    @abstractmethod
    def predict(self, image: Image.Image):
        pass
