from abc import ABC, abstractmethod
from typing import Literal
from PIL import Image


class Img2TextModel(ABC):
    def __init__(
        self,
        cache_dir: str,
        device: Literal["cpu", "cuda", "mps"],
        use_safetensors: bool = True,
    ):
        self.cache_dir = cache_dir
        self.device = device
        self.use_safetensors = use_safetensors

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def unload(self):
        pass

    @abstractmethod
    def predict(self, image: Image.Image):
        pass
