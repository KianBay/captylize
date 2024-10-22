from fastapi import Query, Depends
from captylize.app.dtos.analyses.request import AgeRequest, EmotionRequest, NSFWRequest
from captylize.app.dtos.generations.request import Florence2CaptionRequest


from captylize.ml.models.config import (
    ModelCategory,
    AnalysesType,
    GenerationType,
)
from captylize.ml.manager import model_manager
from captylize.ml.models.caption.advanced.base import AdvancedCaptionModel
from captylize.ml.models.img_to_text_model import Img2TextModel
from captylize.ml.models.caption.basic.base import BasicCaptionModel


async def get_age_model(
    request: AgeRequest = Depends(AgeRequest.as_form),
) -> Img2TextModel[dict[str, float]]:
    return model_manager.get_model(
        ModelCategory.ANALYSES, AnalysesType.AGE, request.model_name
    )


async def get_emotion_model(
    request: EmotionRequest = Depends(EmotionRequest.as_form),
) -> Img2TextModel[dict[str, float]]:
    return model_manager.get_model(
        ModelCategory.ANALYSES, AnalysesType.EMOTION, request.model_name
    )


async def get_nsfw_model(
    request: NSFWRequest = Depends(NSFWRequest.as_form),
) -> Img2TextModel[dict[str, float]]:
    return model_manager.get_model(
        ModelCategory.ANALYSES, AnalysesType.NSFW, request.model_name
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


async def get_florence2_caption_model(
    request: Florence2CaptionRequest = Depends(Florence2CaptionRequest.as_form),
) -> AdvancedCaptionModel:
    model_key = f"florence2_{request.variant}_{request.size}"
    return model_manager.get_model(
        ModelCategory.GENERATION, GenerationType.FLORENCE2_CAPTION, model_key
    )
