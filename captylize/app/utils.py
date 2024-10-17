from fastapi import UploadFile
from PIL import Image
import io
from captylize.app.http_client import async_session


async def _fetch_image_from_url(url: str) -> Image.Image:
    async with async_session.get(url) as response:
        response.raise_for_status()
        return Image.open(io.BytesIO(await response.content.read()))


async def _get_image_from_upload(file: UploadFile) -> Image.Image:
    contents = await file.read()
    return Image.open(io.BytesIO(contents))


async def get_image(
    image_url: str = None, image_file: UploadFile = None
) -> Image.Image:
    if image_url:
        return await _fetch_image_from_url(image_url)
    elif image_file:
        return await _get_image_from_upload(image_file)
