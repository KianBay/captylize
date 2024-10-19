from pydantic import Field

from captylize.app.dtos.shared import InferenceResponse


class CaptionResponse(InferenceResponse):
    caption: str = Field(
        ...,
        description="The generated caption for the image.",
    )

    @classmethod
    def from_prediction(cls, prediction: str, prediction_duration: float):
        return cls(caption=prediction, prediction_duration=prediction_duration)
