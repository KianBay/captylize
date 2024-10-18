from abc import ABC, abstractmethod
from typing import Literal
from PIL import Image

from captylize import logger


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
        self._is_loaded = False

    @abstractmethod
    def _load(self):
        raise NotImplementedError

    @abstractmethod
    def _unload(self):
        raise NotImplementedError

    @abstractmethod
    def _predict(self, image: Image.Image):
        raise NotImplementedError

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self):
        self._load()
        self._is_loaded = True

    def unload(self):
        self._unload()
        self._is_loaded = False

    def predict(self, image: Image.Image):
        if not self._is_loaded:
            logger.debug(f"Model {self.__class__.__name__} is not loaded. Loading...")
            self.load()
        return self._predict(image)
