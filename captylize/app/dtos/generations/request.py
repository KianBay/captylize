from typing import Optional
from pydantic import BaseModel

from captylize.app.dtos.shared import ImageRequest
from captylize.ml.manager import Florence2Task


class BasicCaptionRequest(ImageRequest):
    pass


class Florence2CaptionParams(BaseModel):
    task: Florence2Task
    prompt: Optional[str] = None
