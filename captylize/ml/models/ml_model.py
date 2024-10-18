from abc import ABC, abstractmethod
from types import NotImplementedType
from typing import Literal
from PIL import Image

from captylize.ml.utils.timing import measure_time


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
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def unload(self):
        raise NotImplementedError

    @abstractmethod
    @measure_time
    def predict(self, image: Image.Image):
        raise NotImplementedError
