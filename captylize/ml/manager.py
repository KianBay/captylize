from dataclasses import dataclass
from enum import StrEnum
from typing import Type, Any, Optional, Literal

import torch

from captylize.ml.models.caption.advanced.florence_2 import (
    Florence2FluxModel,
    Florence2PromptGenModel,
    Florence2StandardModel,
)
from captylize.ml.models.caption.basic.vit_gpt2_image_captioning import (
    VitGPT2CaptionModel,
)
from captylize.ml.models.vit_model import ViTImg2TextModel


class ModelCategory(StrEnum):
    ANALYSES = "analyses"
    GENERATION = "generation"


class AnalysesType(StrEnum):
    AGE = "age"
    EMOTION = "emotion"
    NSFW = "nsfw"


class GenerationType(StrEnum):
    BASIC_CAPTION = "basic_caption"
    ADVANCED_CAPTION = "advanced_caption"


ModelType = AnalysesType | GenerationType


@dataclass
class ModelInfo:
    name: str
    path: str
    model_class: Type[Any]
    is_default: bool = False


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
    ):
        if category not in self.registry:
            self.registry[category] = {}
        if model_type not in self.registry[category]:
            self.registry[category][model_type] = {}
        self.registry[category][model_type][name] = ModelInfo(
            name=name, path=path, model_class=model_class, is_default=is_default
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

# ANALYSES REGISTRATION

model_manager.register_model(
    ModelCategory.ANALYSES,
    AnalysesType.AGE,
    "vit_age_classifier",
    "nateraw/vit-age-classifier",
    ViTImg2TextModel,
    is_default=True,
)
model_manager.register_model(
    ModelCategory.ANALYSES,
    AnalysesType.EMOTION,
    "vit_emotion_classifier",
    "dima806/facial_emotions_image_detection",
    ViTImg2TextModel,
    is_default=True,
)
model_manager.register_model(
    ModelCategory.ANALYSES,
    AnalysesType.NSFW,
    "vit_nsfw_detector",
    "AdamCodd/vit-base-nsfw-detector",
    ViTImg2TextModel,
    is_default=True,
)
# BASIC CAPTION REGISTRATION

model_manager.register_model(
    ModelCategory.GENERATION,
    GenerationType.BASIC_CAPTION,
    "vit_gpt2_image_captioning",
    "nlpconnect/vit-gpt2-image-captioning",
    VitGPT2CaptionModel,
    is_default=True,
)

# ADVANCED CAPTION REGISTRATION

model_manager.register_model(
    ModelCategory.GENERATION,
    GenerationType.ADVANCED_CAPTION,
    "florence2_large",
    "microsoft/Florence-2-large",
    Florence2StandardModel,
    is_default=True,
)

model_manager.register_model(
    ModelCategory.GENERATION,
    GenerationType.ADVANCED_CAPTION,
    "florence2_base",
    "microsoft/Florence-2-base",
    Florence2StandardModel,
)

model_manager.register_model(
    ModelCategory.GENERATION,
    GenerationType.ADVANCED_CAPTION,
    "florence2_promptgen_large",
    "MiaoshouAI/Florence-2-large-PromptGen-v1.5",
    Florence2PromptGenModel,
)

model_manager.register_model(
    ModelCategory.GENERATION,
    GenerationType.ADVANCED_CAPTION,
    "florence2_promptgen_base",
    "MiaoshouAI/Florence-2-base-PromptGen-v1.5",
    Florence2PromptGenModel,
)

model_manager.register_model(
    ModelCategory.GENERATION,
    GenerationType.ADVANCED_CAPTION,
    "florence2_flux_large",
    "gokaygokay/Florence-2-Flux-Large",
    Florence2FluxModel,
)

model_manager.register_model(
    ModelCategory.GENERATION,
    GenerationType.ADVANCED_CAPTION,
    "florence2_flux_base",
    "gokaygokay/Florence-2-Flux",
    Florence2FluxModel,
)
