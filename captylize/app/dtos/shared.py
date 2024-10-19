from fastapi import UploadFile
from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import Optional

ALLOWED_IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]


class InferenceResponse(BaseModel):
    prediction_duration: float = Field(
        ..., description="The duration of the prediction in milliseconds"
    )


class ImageRequest(BaseModel):
    """
    Accepts either an image URL or an image file, NOT both.
    """

    image_url: Optional[HttpUrl] = Field(
        None,
        description=f"The URL of the image to analyze, must resolve to a {ALLOWED_IMAGE_EXTENSIONS} file.",
    )
    image_file: Optional[UploadFile] = Field(
        None,
        description=f"The image file to analyze, must be of type {ALLOWED_IMAGE_EXTENSIONS}.",
    )

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
