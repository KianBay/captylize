from typing import Literal
from pydantic import BaseModel

from captylize.app.dtos.shared import InferenceResponse

AgeRange = Literal[
    "0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"
]


Emotion = Literal["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


# Nsfw categories are a todo for now,
# Some models classify as sfw/nsfw, others as normal/nsfw, others as normal/porn/sexy etc
# For now we stick to easily castable of FalconsAI & AdamCobb versions, which are normal/nsfw and sfw/nsfw respectively

NsfwCategory = Literal["sfw", "nsfw", "normal"]


class AgeResponse(InferenceResponse):
    predicted_age: AgeRange
    probabilities: dict[AgeRange, float]

    @classmethod
    def from_prediction(
        cls, prediction: dict[AgeRange, float], prediction_duration: float
    ):
        predicted_age = max(prediction, key=prediction.get)
        return cls(
            predicted_age=predicted_age,
            probabilities=prediction,
            prediction_duration=prediction_duration,
        )


class EmotionResponse(InferenceResponse):
    predicted_emotion: Emotion
    probabilities: dict[Emotion, float]

    @classmethod
    def from_prediction(
        cls, prediction: dict[Emotion, float], prediction_duration: float
    ):
        predicted_emotion = max(prediction, key=prediction.get)
        return cls(
            predicted_emotion=predicted_emotion,
            probabilities=prediction,
            prediction_duration=prediction_duration,
        )


class NSFWResponse(InferenceResponse):
    predicted_category: NsfwCategory
    probabilities: dict[NsfwCategory, float]

    @classmethod
    def from_prediction(
        cls, prediction: dict[NsfwCategory, float], prediction_duration: float
    ):
        predicted_category = max(prediction, key=prediction.get)
        return cls(
            predicted_category=predicted_category,
            probabilities=prediction,
            prediction_duration=prediction_duration,
        )
