from pydantic import Field
from fastapi import Form, File, UploadFile
from typing import Optional
from captylize.app.dtos.shared import BaseImageRequest
from captylize.ml.manager import Florence2Task
from captylize.ml.models.config import (
    Florence2Variant,
    Florence2Size,
    VITCaptionModelName,
)


class BasicCaptionRequest(BaseImageRequest):
    model_name: VITCaptionModelName = Field(
        default=VITCaptionModelName.VIT_GPT2_IMAGE_CAPTIONING,
        description="The name of the model to use.",
    )


class Florence2CaptionRequest(BaseImageRequest):
    task: Florence2Task = Field(..., description="The task for Florence2 model.")
    variant: Florence2Variant = Field(
        default=Florence2Variant.STANDARD, description="The Florence-2 variant to use."
    )
    size: Florence2Size = Field(
        default=Florence2Size.BASE, description="The size of the model to use."
    )

    @classmethod
    def as_form(
        cls,
        task: Florence2Task = Form(...),
        variant: Florence2Variant = Form(default=Florence2Variant.STANDARD),
        size: Florence2Size = Form(default=Florence2Size.BASE),
        image_url: Optional[str] = Form(None),
        image_file: Optional[UploadFile] = File(None),
    ):
        return cls(
            task=task,
            variant=variant,
            size=size,
            image_url=image_url,
            image_file=image_file,
        )
