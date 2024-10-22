from fastapi import UploadFile, Form, File
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional

ALLOWED_IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]


class InferenceResponse(BaseModel):
    prediction_duration: float = Field(
        ..., description="The duration of the prediction in milliseconds"
    )


class BaseImageRequest(BaseModel):
    image_url: Optional[HttpUrl] = Field(None)
    image_file: Optional[UploadFile] = Field(None)

    @classmethod
    def as_form(
        cls,
        image_url: Optional[HttpUrl] = Form(
            None, description="The URL of the image to analyze."
        ),
        image_file: Optional[UploadFile] = File(
            None, description="The image file to analyze."
        ),
    ):
        return cls(image_url=image_url, image_file=image_file)
