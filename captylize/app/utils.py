from fastapi import UploadFile
from PIL import Image
import io
from captylize.app.http_client import async_session
from pydantic import HttpUrl
from typing import Optional


async def _fetch_image_from_url(url: str) -> Image.Image:
    response = await async_session.get(url)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


async def _get_image_from_upload(file: UploadFile) -> Image.Image:
    contents = await file.read()
    return Image.open(io.BytesIO(contents))


async def get_image(
    image_url: Optional[HttpUrl] = None, image_file: Optional[UploadFile] = None
) -> Image.Image:
    if image_url:
        url_str = str(image_url)
        return await _fetch_image_from_url(url_str)
    elif image_file:
        return await _get_image_from_upload(image_file)
    else:
        raise ValueError("Either image_url or image_file must be provided.")
