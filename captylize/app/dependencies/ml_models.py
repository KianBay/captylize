from fastapi import Query
from typing import Optional
from captylize.app.dtos.generations.request import Florence2CaptionParams
from captylize.ml.manager import (
    model_manager,
    ModelCategory,
    AnalysesType,
    GenerationType,
)
from captylize.ml.models.caption.advanced.base import AdvancedCaptionModel
from captylize.ml.models.img_to_text_model import Img2TextModel
from captylize.ml.models.caption.basic.base import BasicCaptionModel


async def get_age_model(
    model_name: str = Query(
        None,
        description="The name of the model to use. Can be left empty to use the default model.",
    ),
) -> Img2TextModel[dict[str, float]]:
    return model_manager.get_model(ModelCategory.ANALYSES, AnalysesType.AGE, model_name)


async def get_emotion_model(
    model_name: str = Query(
        None,
        description="The name of the model to use. Can be left empty to use the default model.",
    ),
) -> Img2TextModel[dict[str, float]]:
    return model_manager.get_model(
        ModelCategory.ANALYSES, AnalysesType.EMOTION, model_name
    )


async def get_nsfw_model(
    model_name: str = Query(
        None,
        description="The name of the model to use. Can be left empty to use the default model.",
    ),
) -> Img2TextModel[dict[str, float]]:
    return model_manager.get_model(
        ModelCategory.ANALYSES, AnalysesType.NSFW, model_name
    )


async def get_vit_caption_model(
    model_name: str = Query(
        None,
        description="The name of the model to use. Can be left empty to use the default model.",
    ),
) -> BasicCaptionModel:
    return model_manager.get_model(
        ModelCategory.GENERATION, GenerationType.VIT_CAPTION, model_name
    )


async def get_florence2_caption_params(
    task: Optional[str] = Query(
        None,
        description="The task to use the model for. Available tasks depend on specific model - check docs.",
    ),
    prompt: Optional[str] = Query(
        None,
        description="Prompt to guide the model's caption generation. Can be left empty to use the default prompt.",
    ),
) -> Florence2CaptionParams:
    return Florence2CaptionParams(task=task, prompt=prompt)


async def get_florence2_caption_model(
    version: str = Query(
        "standard",
        description="The Florence-2 variant to use: 'standard', 'promptgen' or 'flux'.",
    ),
    size: str = Query(
        "base",
        description="The size of the model to use: 'base' or 'large'.",
    ),
) -> AdvancedCaptionModel:
    model_key = f"florence2_{version}_{size}"
    return model_manager.get_model(
        ModelCategory.GENERATION, GenerationType.FLORENCE2_CAPTION, model_key
    )
