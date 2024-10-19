from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ValidationError


from captylize.app.dependencies.ml_models import (
    get_florence2_caption_model,
    get_florence2_caption_params,
    get_vit_caption_model,
)
from captylize.app.dtos.generations.request import (
    Florence2CaptionParams,
)
from captylize.app.dtos.generations.response import CaptionResponse
from captylize.app.dtos.shared import ImageRequest
from captylize.app.routers.shared import validate_image_input
from captylize.app.utils import get_image
from captylize.logger import get_logger


router = APIRouter(prefix="/generations")

logger = get_logger(__name__)


@router.post("/captions/vit")
async def create_vit_caption(
    image_input: ImageRequest = Depends(validate_image_input),
    caption_model=Depends(get_vit_caption_model),
) -> CaptionResponse:
    try:
        image = await get_image(image_input.image_url, image_input.image_file)
        result, duration = caption_model.predict(image)
        return CaptionResponse.from_prediction(
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


@router.post("/captions/florence-2")
async def create_florence_2_caption(
    image_input: ImageRequest = Depends(validate_image_input),
    caption_params: Florence2CaptionParams = Depends(get_florence2_caption_params),
    caption_model=Depends(get_florence2_caption_model),
) -> CaptionResponse:
    try:
        image = await get_image(image_input.image_url, image_input.image_file)
        result, duration = caption_model.predict(
            image, task=caption_params.task, prompt=caption_params.prompt
        )
        return CaptionResponse.from_prediction(
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
