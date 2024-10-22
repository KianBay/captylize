from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from captylize.app.dtos.analyses.response import (
    AgeResponse,
    EmotionResponse,
    NSFWResponse,
)
from captylize.logger import get_logger
from captylize.app.dtos.analyses.request import AgeRequest, EmotionRequest, NSFWRequest
from captylize.app.utils import get_image
from captylize.app.dependencies.ml_models import (
    get_age_model,
    get_emotion_model,
    get_nsfw_model,
)
from captylize.ml.models.img_to_text_model import Img2TextModel

router = APIRouter(prefix="/analyses", tags=["analyses"])

logger = get_logger(__name__)


@router.post("/ages")
async def create_age_analysis(
    request: AgeRequest = Depends(AgeRequest.as_form),
    age_model: Img2TextModel[dict[str, float]] = Depends(get_age_model),
) -> AgeResponse:
    logger.info("Creating age analysis")
    try:
        image = await get_image(request)
        result, duration = age_model.predict(image)
        return AgeResponse.from_prediction(
            prediction=result, prediction_duration=duration
        )
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emotions")
async def create_emotion_analysis(
    request: EmotionRequest = Depends(EmotionRequest.as_form),
    emotion_model: Img2TextModel[dict[str, float]] = Depends(get_emotion_model),
) -> EmotionResponse:
    try:
        image = await get_image(request)
        result, duration = emotion_model.predict(image)
        return EmotionResponse.from_prediction(
            prediction=result, prediction_duration=duration
        )
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nsfw")
async def create_nsfw_analysis(
    request: NSFWRequest = Depends(NSFWRequest.as_form),
    nsfw_model: Img2TextModel[dict[str, float]] = Depends(get_nsfw_model),
) -> NSFWResponse:
    try:
        image = await get_image(request)
        result, duration = nsfw_model.predict(image)
        return NSFWResponse.from_prediction(
            prediction=result, prediction_duration=duration
        )
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
