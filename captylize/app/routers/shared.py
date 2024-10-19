from fastapi import Form, File, UploadFile
from pydantic import ValidationError
from typing import Optional

from captylize.app.dtos.shared import ImageRequest, ALLOWED_IMAGE_EXTENSIONS


async def validate_image_input(
    image_url: Optional[str] = Form(
        default=None,
        description=f"The URL of the image to analyze. Must resolve to a {ALLOWED_IMAGE_EXTENSIONS} file.",
    ),
    image_file: Optional[UploadFile] = File(
        default=None,
        description=f"The image file to analyze. Must be of type {ALLOWED_IMAGE_EXTENSIONS}.",
    ),
) -> ImageRequest:
    """
    Validates the input for image analysis.
    Accepts either an image URL or an uploaded image file, but not both.
    """
    if image_file and image_file.filename and image_url:
        raise ValueError("Provide either image_url or image_file, not both.")

    if image_file and image_file.filename:
        if not any(
            image_file.filename.lower().endswith(ext)
            for ext in ALLOWED_IMAGE_EXTENSIONS
        ):
            raise ValidationError(
                f"Invalid image file type. Must be one of {ALLOWED_IMAGE_EXTENSIONS}"
            )
        return ImageRequest(image_file=image_file)
    elif image_url:
        return ImageRequest(image_url=image_url)
    else:
        raise ValueError("Either image_url or image_file must be provided.")
