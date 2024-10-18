from abc import ABC, abstractmethod
from typing import Literal, TypeVar, Generic
from PIL import Image

from captylize import logger
from captylize.ml.utils.timing import measure_time

T = TypeVar("T")


class Img2TextModel(ABC, Generic[T]):
    def __init__(
        self,
        cache_dir: str,
        device: Literal["cpu", "cuda", "mps"],
        use_safetensors: bool = True,
    ):
        self.cache_dir = cache_dir
        self.device = device
        self.use_safetensors = use_safetensors
        self._is_loaded: bool = False

    @abstractmethod
    def _load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _unload(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict(self, image: Image.Image) -> T:
        raise NotImplementedError

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        self._load()
        self._is_loaded = True

    def unload(self) -> None:
        self._unload()
        self._is_loaded = False

    @measure_time
    def predict(self, image: Image.Image) -> T:
        if not self._is_loaded:
            logger.debug(f"Model {self.__class__.__name__} is not loaded. Loading...")
            self.load()
        return self._predict(image)
