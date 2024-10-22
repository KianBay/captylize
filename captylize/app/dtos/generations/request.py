from typing import Optional
from pydantic import BaseModel, Field, HttpUrl
from fastapi import UploadFile, Form
from captylize.app.dtos.shared import BaseImageRequest
from captylize.ml.manager import Florence2Task


class BasicCaptionRequest(BaseImageRequest):
    pass


class Florence2CaptionRequest(BaseImageRequest):
    task: Florence2Task = Field(..., description="The task for Florence2 model.")
    prompt: Optional[str] = Field(None, description="Optional prompt for the task.")

    @classmethod
    def as_form(
        cls,
        image_url: Optional[HttpUrl] = Form(None),
        image_file: Optional[UploadFile] = Form(None),
        task: Florence2Task = Form(...),
        prompt: Optional[str] = Form(None),
    ):
        return cls(image_url=image_url, image_file=image_file, task=task, prompt=prompt)
