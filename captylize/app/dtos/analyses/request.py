from pydantic import Field
from captylize.app.dtos.shared import BaseImageRequest
from captylize.ml.models.config import AgeModelName, EmotionModelName, NSFWModelName


class AgeRequest(BaseImageRequest):
    model_name: AgeModelName = Field(
        default=AgeModelName.VIT_AGE_CLASSIFIER,
        description="The name of the model to use.",
    )


class EmotionRequest(BaseImageRequest):
    model_name: EmotionModelName = Field(
        default=EmotionModelName.VIT_EMOTION_CLASSIFIER,
        description="The name of the model to use.",
    )


class NSFWRequest(BaseImageRequest):
    model_name: NSFWModelName = Field(
        default=NSFWModelName.VIT_NSFW_DETECTOR,
        description="The name of the model to use.",
    )
