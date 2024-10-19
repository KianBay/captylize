from typing import Literal
from pydantic import BaseModel, Field

from captylize.app.dtos.shared import InferenceResponse

AgeRange = Literal[
    "0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"
]


class AgeRangeProb(BaseModel):
    age_range: AgeRange
    probability: float


Emotion = Literal["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


class EmotionProb(BaseModel):
    emotion: Emotion
    probability: float


class AgeResponse(InferenceResponse):
    predicted_age: AgeRange
    probabilities: list[AgeRangeProb]


# Nsfw categories are a todo for now,
# Some models classify as sfw/nsfw, others as normal/nsfw, others as normal/porn/sexy etc
# For now we stick to easily castable of FalconsAI & AdamCobb versions, which are normal/nsfw and sfw/nsfw respectively

NsfwCategory = Literal["sfw", "nsfw", "normal"]


class NsfwProb(BaseModel):
    category: NsfwCategory
    probability: float

    @classmethod
    def from_prediction(cls, prediction: dict[str, float], prediction_duration: float):
        predicted_age = max(prediction, key=prediction.get)
        return cls(
            predicted_age=predicted_age,
            probabilities=prediction,
            prediction_duration=prediction_duration,
        )


class EmotionResponse(InferenceResponse):
    predicted_emotion: Emotion
    probabilities: list[EmotionProb]

    @classmethod
    def from_prediction(cls, prediction: dict[str, float], prediction_duration: float):
        predicted_emotion = max(prediction, key=prediction.get)
        return cls(
            predicted_emotion=predicted_emotion,
            probabilities=prediction,
            prediction_duration=prediction_duration,
        )


class NSFWResponse(InferenceResponse):
    predicted_category: NsfwCategory
    probabilities: list[NsfwProb]

    @classmethod
    def from_prediction(cls, prediction: dict[str, float], prediction_duration: float):
        predicted_category = max(prediction, key=prediction.get)
        return cls(
            predicted_category=predicted_category,
            probabilities=prediction,
            prediction_duration=prediction_duration,
        )
