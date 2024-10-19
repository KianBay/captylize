from fastapi import Query
from captylize.ml.manager import (
    model_manager,
    ModelCategory,
    AnalysesType,
    GenerationType,
)
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


async def get_basic_caption_model(
    model_name: str = Query(
        None,
        description="The name of the model to use. Can be left empty to use the default model.",
    ),
) -> BasicCaptionModel:
    return model_manager.get_model(
        ModelCategory.GENERATION, GenerationType.BASIC_CAPTION, model_name
    )
