from abc import ABC, abstractmethod
from typing import Literal, TypeVar, Generic
from PIL import Image

from captylize.logger import get_logger
from captylize.ml.utils.timing import measure_time

T = TypeVar("T")

logger = get_logger(__name__)


class Img2TextModel(ABC, Generic[T]):
    def __init__(
        self,
        model_name: str,
        model_location: str,
        cache_dir: str,
        device: Literal["cpu", "cuda", "mps"],
        use_safetensors: bool = True,
    ):
        self.model_name = model_name
        self.model_location = model_location
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
            logger.debug(f"Model {self.model_name} is not loaded. Loading...")
            self.load()
        return self._predict(image)
