from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from captylize.app.dependencies.ml_models import get_basic_caption_model
from captylize.app.dtos.generations.request import BasicCaptionRequest
from captylize.app.dtos.generations.response import BasicCaptionResponse
from captylize.app.routers.shared import validate_image_input
from captylize.app.utils import get_image
from captylize.logger import get_logger
from captylize.ml.models.img_to_text_model import Img2TextModel


router = APIRouter(prefix="/generations")

logger = get_logger(__name__)


@router.post("/captions/basic")
async def create_basic_caption(
    image_input: BasicCaptionRequest = Depends(validate_image_input),
    caption_model: Img2TextModel[str] = Depends(get_basic_caption_model),
) -> BasicCaptionResponse:
    try:
        image = await get_image(image_input.image_url, image_input.image_file)
        result, duration = caption_model.predict(image)
        return BasicCaptionResponse.from_prediction(
            prediction=result, prediction_duration=duration
        )
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        raise HTTPException(status_code=500, detail=str(e))
