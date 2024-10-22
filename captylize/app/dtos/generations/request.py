from typing import Optional
from pydantic import Field, HttpUrl
from fastapi import UploadFile, Form, File
from captylize.app.dtos.shared import BaseImageRequest
from captylize.ml.manager import Florence2Task


class BasicCaptionRequest(BaseImageRequest):
    pass


class Florence2CaptionRequest(BaseImageRequest):
    task: Florence2Task = Field(..., description="The task for Florence2 model.")

    @classmethod
    def as_form(
        cls,
        image_url: Optional[HttpUrl] = Form(None),
        image_file: Optional[UploadFile] = File(None),
        task: Florence2Task = Form(...),
    ):
        return cls(image_url=image_url, image_file=image_file, task=task)
