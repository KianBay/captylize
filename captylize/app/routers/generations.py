from fastapi import APIRouter, Depends, HTTPException, Form, File, UploadFile
from pydantic import ValidationError
from typing import Optional


from captylize.app.dependencies.ml_models import (
    get_florence2_caption_model,
    get_vit_caption_model,
)
from captylize.app.dtos.generations.request import (
    BasicCaptionRequest,
    Florence2CaptionRequest,
)
from captylize.app.dtos.generations.response import CaptionResponse
from captylize.app.utils import get_image
from captylize.logger import get_logger
from captylize.ml.models.caption.advanced.base import AdvancedCaptionModel
from captylize.ml.models.config import Florence2Task


router = APIRouter(prefix="/generations")

logger = get_logger(__name__)


@router.post("/captions/vit")
async def create_vit_caption(
    request: BasicCaptionRequest = Depends(BasicCaptionRequest.as_form),
    caption_model=Depends(get_vit_caption_model),
) -> CaptionResponse:
    try:
        image = await get_image(request)
        result, duration = caption_model.predict(image)
        return CaptionResponse.from_prediction(
            prediction=result, prediction_duration=duration
        )
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/captions/florence-2")
async def create_florence_2_caption(
    task: Florence2Task = Form(...),
    image_url: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
    caption_model: AdvancedCaptionModel = Depends(get_florence2_caption_model),
) -> CaptionResponse:
    request = Florence2CaptionRequest(
        task=task, image_url=image_url, image_file=image_file
    )
    try:
        image = await get_image(request)
        result, duration = caption_model.predict(image, task=request.task)
        return CaptionResponse.from_prediction(
            prediction=result, prediction_duration=duration
        )
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        raise HTTPException(status_code=500, detail=str(e))
