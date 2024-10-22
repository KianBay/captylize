from pydantic import Field
from fastapi import Form
from captylize.app.dtos.shared import BaseImageRequest
from captylize.ml.manager import Florence2Task


class BasicCaptionRequest(BaseImageRequest):
    pass


class Florence2CaptionRequest(BaseImageRequest):
    task: Florence2Task = Field(..., description="The task for Florence2 model.")

    @classmethod
    def as_form(cls, task: Florence2Task = Form(...), **kwargs):
        base_request = super().as_form(**kwargs)
        return cls(**base_request.model_dump(), task=task)
