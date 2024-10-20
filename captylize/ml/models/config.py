from dataclasses import dataclass, field
from enum import StrEnum
from typing import Type, Any, Optional


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
