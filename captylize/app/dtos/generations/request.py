from typing import Optional
from pydantic import BaseModel, Field
from captylize.app.dtos.shared import ImageRequest


class BasicCaptionRequest(ImageRequest):
    pass


class Florence2CaptionParams(BaseModel):
    task: Optional[str] = Field(
        None,
        description="The task to use the model for. Available tasks depend on specific model - check docs.",
    )
    prompt: Optional[str] = Field(
        None,
        description="Prompt to guide the model's caption generation. Can be left empty to use the default prompt.",
    )
