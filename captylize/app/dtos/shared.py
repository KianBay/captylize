from fastapi import UploadFile
from pydantic import BaseModel, HttpUrl, field_validator
from typing import Optional

ALLOWED_IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]


class ImageRequest(BaseModel):
    image_url: Optional[HttpUrl] = None
    image_file: Optional[UploadFile] = None

    @field_validator("image_url")
    def validate_image_url(cls, v: HttpUrl):
        if v:
            if not any(
                v.path.lower().endswith(f".{ext}") for ext in ALLOWED_IMAGE_EXTENSIONS
            ):
                raise ValueError(
                    f"Invalid image URL extension. Must be one of {ALLOWED_IMAGE_EXTENSIONS}"
                )
        return v
