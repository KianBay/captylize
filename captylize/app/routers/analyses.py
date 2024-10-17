from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from captylize.app.dtos.analyses.response import AgeResponse
from captylize.logger import get_logger
from captylize.app.dtos.analyses.request import AgeRequest, MoodRequest, NSFWRequest
from captylize.app.routers.shared import validate_image_input
from captylize.app.utils import get_image
from captylize.app.dependencies.ml_models import get_age_model
from captylize.ml.models.ml_model import Img2TextModel

router = APIRouter(prefix="/analyses", tags=["analyses"])

logger = get_logger(__name__)


@router.post("/ages")
async def create_age_analysis(
    image_input: AgeRequest = Depends(validate_image_input),
    age_model: Img2TextModel = Depends(get_age_model),
):
    logger.info("Creating age analysis")
    try:
        image = await get_image(image_input.image_url, image_input.image_file)
        result = age_model.predict(image)
        return AgeResponse.from_prediction(result)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/moods")
async def create_mood_analysis(
    image_input: MoodRequest = Depends(validate_image_input),
):
    try:
        image = await get_image(image_input.image_url, image_input.image_file)
        return {"width": image.width, "height": image.height}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nsfw")
async def create_nsfw_analysis(
    image_input: NSFWRequest = Depends(validate_image_input),
):
    try:
        image = await get_image(image_input.image_url, image_input.image_file)
        return {"width": image.width, "height": image.height}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
