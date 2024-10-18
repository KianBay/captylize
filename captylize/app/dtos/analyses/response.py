from typing import Literal
from pydantic import BaseModel, Field

from captylize.app.dtos.shared import InferenceResponse

AgeRange = Literal[
    "0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"
]


class AgeResponse(InferenceResponse):
    predicted_age: AgeRange
    probabilities: dict[str, float]

    @classmethod
    def from_prediction(cls, prediction: dict[str, float], prediction_duration: float):
        predicted_age = max(prediction, key=prediction.get)
        return cls(
            predicted_age=predicted_age,
            probabilities=prediction,
            prediction_duration=prediction_duration,
        )


class EmotionResponse(InferenceResponse):
    predicted_emotion: str
    probabilities: dict[str, float]

    @classmethod
    def from_prediction(cls, prediction: dict[str, float], prediction_duration: float):
        predicted_emotion = max(prediction, key=prediction.get)
        return cls(
            predicted_emotion=predicted_emotion,
            probabilities=prediction,
            prediction_duration=prediction_duration,
        )
