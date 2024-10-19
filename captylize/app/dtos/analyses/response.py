from enum import StrEnum

from pydantic import Field

from captylize.app.dtos.shared import InferenceResponse


class AgeRange(StrEnum):
    AGE_0_2 = "0-2"
    AGE_3_9 = "3-9"
    AGE_10_19 = "10-19"
    AGE_20_29 = "20-29"
    AGE_30_39 = "30-39"
    AGE_40_49 = "40-49"
    AGE_50_59 = "50-59"
    AGE_60_69 = "60-69"
    AGE_70_PLUS = "70+"


class Emotion(StrEnum):
    ANGRY = "angry"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"


# Nsfw categories are a todo for now,
# Some models classify as sfw/nsfw, others as normal/nsfw, others as normal/porn/sexy etc
# For now we stick to easily castable of FalconsAI & AdamCobb versions, which are normal/nsfw and sfw/nsfw respectively
class NsfwCategory(StrEnum):
    SFW = "sfw"
    NSFW = "nsfw"


class AgeResponse(InferenceResponse):
    predicted_age: AgeRange = Field(
        ...,
        description="The predicted age range, i.e. age range with highest probability.",
    )
    probabilities: dict[AgeRange, float] = Field(
        ...,
        description="The probabilities of each age range",
        example={age_range: 0.0 for age_range in AgeRange},
    )

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
    predicted_emotion: Emotion = Field(
        ...,
        description="The predicted emotion, i.e. emotion with highest probability.",
    )
    probabilities: dict[Emotion, float] = Field(
        ...,
        description="The probabilities of each emotion",
        example={emotion: 0.0 for emotion in Emotion},
    )

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
    predicted_category: NsfwCategory = Field(
        ...,
        description="The predicted NSFW category - exact fields may vary by model; check documentation for specific model. Always expect sfw and nsfw.",
    )
    probabilities: dict[NsfwCategory, float] = Field(
        ...,
        description="The probabilities of each NSFW category - exact fields may vary by model; check documentation for specific model. Always expect sfw and nsfw.",
        example={category: 0.0 for category in NsfwCategory},
    )

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
