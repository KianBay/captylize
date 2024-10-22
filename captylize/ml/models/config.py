from dataclasses import dataclass, field
from enum import StrEnum
from typing import Type, Any, Optional


class AgeModelName(StrEnum):
    VIT_AGE_CLASSIFIER = "vit_age_classifier"


class EmotionModelName(StrEnum):
    VIT_EMOTION_CLASSIFIER = "vit_emotion_classifier"


class NSFWModelName(StrEnum):
    VIT_NSFW_DETECTOR = "vit_nsfw_detector"


class VITCaptionModelName(StrEnum):
    VIT_GPT2_IMAGE_CAPTIONING = "vit_gpt2_image_captioning"


class Florence2ModelName(StrEnum):
    STANDARD_LARGE = "florence2_standard_large"
    STANDARD_BASE = "florence2_standard_base"
    PROMPTGEN_LARGE = "florence2_promptgen_large"
    PROMPTGEN_BASE = "florence2_promptgen_base"
    FLUX_LARGE = "florence2_flux_large"
    FLUX_BASE = "florence2_flux_base"


class ModelCategory(StrEnum):
    ANALYSES = "analyses"
    GENERATION = "generation"


class AnalysesType(StrEnum):
    AGE = "age"
    EMOTION = "emotion"
    NSFW = "nsfw"


class GenerationType(StrEnum):
    VIT_CAPTION = "vit_caption"
    FLORENCE2_CAPTION = "florence2_caption"


ModelType = AnalysesType | GenerationType


class Florence2Task(StrEnum):
    CAPTION = "caption"
    DETAILED_CAPTION = "detailed_caption"
    MORE_DETAILED_CAPTION = "more_detailed_caption"
    GENERATE_TAGS = "generate_tags"
    MIXED_CAPTION = "mixed_caption"
    DESCRIPTION = "description"


class Florence2Variant(StrEnum):
    STANDARD = "standard"
    PROMPTGEN = "promptgen"
    FLUX = "flux"


class Florence2Size(StrEnum):
    BASE = "base"
    LARGE = "large"


@dataclass
class ModelInfo:
    name: str
    path: str
    model_class: Type[Any]
    is_default: bool = False
    available_tasks: list[Florence2Task] = field(default_factory=list)
    default_task: Optional[Florence2Task] = None
