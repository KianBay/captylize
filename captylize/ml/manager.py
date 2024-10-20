from typing import Type, Any, Optional, Literal

import torch

from captylize.ml.models.config import (
    ModelCategory,
    ModelType,
    ModelInfo,
    Florence2Task,
)

from captylize.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    def __init__(
        self,
        cache_dir: str = "./model_cache",
        device: Optional[Literal["cpu", "cuda", "mps"]] = None,
    ):
        if device is None:
            self.device = self._get_device()
        else:
            self.device = device

        self.cache_dir = cache_dir
        self.registry: dict[ModelCategory, dict[ModelType, dict[str, ModelInfo]]] = {
            category: {} for category in ModelCategory
        }
        self.loaded_models: dict[ModelCategory, dict[ModelType, dict[str, Any]]] = {
            category: {} for category in ModelCategory
        }

    def _get_device(self) -> Literal["cpu", "cuda", "mps"]:
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def register_model(
        self,
        category: ModelCategory,
        model_type: ModelType,
        name: str,
        path: str,
        model_class: Type[Any],
        is_default: bool = False,
        available_tasks: Optional[list[Florence2Task]] = None,
        default_task: Optional[Florence2Task] = None,
    ):
        if category not in self.registry:
            logger.info(f"Register: category: {category}")
            self.registry[category] = {}
        if model_type not in self.registry[category]:
            logger.info(f"Register: type: {model_type}, category: {category}")
            self.registry[category][model_type] = {}
        logger.info(f"Register: {name}, type: {model_type}, category: {category}")
        self.registry[category][model_type][name] = ModelInfo(
            name=name,
            path=path,
            model_class=model_class,
            is_default=is_default,
            available_tasks=available_tasks or [],
            default_task=default_task,
        )

    def get_model(
        self, category: ModelCategory, model_type: ModelType, name: Optional[str] = None
    ):
        if name is None:
            name = next(
                model_name
                for model_name, model_info in self.registry[category][
                    model_type
                ].items()
                if model_info.is_default
            )
        if (
            model_type not in self.loaded_models[category]
            or name not in self.loaded_models[category][model_type]
        ):
            self._load_model(category, model_type, name)

        return self.loaded_models[category][model_type][name]

    def _load_model(self, category: ModelCategory, model_type: ModelType, name: str):
        model_info = self.registry[category][model_type][name]
        if model_type not in self.loaded_models[category]:
            self.loaded_models[category][model_type] = {}
        self.loaded_models[category][model_type][name] = model_info.model_class(
            model_name=model_info.name,
            model_location=model_info.path,
            cache_dir=self.cache_dir,
            device=self.device,
        )
        logger.info(f"Load: {name}, type: {model_type}, category: {category}")
        self.loaded_models[category][model_type][name].load()

    def load_default_models(self):
        for category in self.registry:
            for model_type in self.registry[category]:
                default_model = next(
                    (
                        name
                        for name, info in self.registry[category][model_type].items()
                        if info.is_default
                    ),
                    None,
                )
                if default_model:
                    self._load_model(category, model_type, default_model)

    def unload_all_models(self):
        for category in self.loaded_models.values():
            for model_type in category.values():
                for model in model_type.values():
                    model.unload()
        self.loaded_models = {category: {} for category in ModelCategory}


model_manager = ModelManager(cache_dir="./model_cache")
