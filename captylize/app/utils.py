from fastapi import UploadFile
from PIL import Image
import io
from captylize.app.dtos.shared import BaseImageRequest
from captylize.app.http_client import async_session

import logging

logger = logging.getLogger(__name__)


async def _fetch_image_from_url(url: str) -> Image.Image:
    response = await async_session.get(url)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


async def _get_image_from_upload(file: UploadFile) -> Image.Image:
    contents = await file.read()
    logger.info(f"File contents size: {len(contents)} bytes")
    try:
        return Image.open(io.BytesIO(contents))
    except Exception as e:
        logger.error(f"Error opening image: {str(e)}")
        raise ValueError("Invalid image file")


async def get_image(request: BaseImageRequest) -> Image.Image:
    if request.image_url:
        url_str = str(request.image_url)
        return await _fetch_image_from_url(url_str)
    elif request.image_file:
        logger.info(f"Received file: {request.image_file.filename}")
        return await _get_image_from_upload(request.image_file)
    else:
        logger.error("Either image_url or image_file must be provided.")
        raise ValueError("Either image_url or image_file must be provided.")
