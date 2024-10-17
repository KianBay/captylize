from abc import ABC, abstractmethod
from PIL import Image


class Img2TextModel(ABC):
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def unload(self):
        pass

    @abstractmethod
    def predict(self, image: Image.Image):
        pass
