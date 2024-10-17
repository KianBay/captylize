from typing import Literal
from pydantic import BaseModel, Field

AgeRange = Literal[
    "0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"
]


class AgeProbabilities(BaseModel):
    """0-2": float
    "3-9": float
    "10-19": float
    "20-29": float
    "30-39": float
    "40-49": float
    "50-59": float
    "60-69": float
    "70+": float"""


class AgeResponse(BaseModel):
    predicted_age: AgeRange = Field(
        ..., description="The predicted age range with the highest probability"
    )
    probabilities: AgeProbabilities = Field(
        ..., description="Probabilities for each age range"
    )

    @classmethod
    def from_prediction(cls, prediction: dict[str, float]):
        # Convert "more than 70" to "70+"
        if "more than 70" in prediction:
            prediction["70+"] = prediction.pop("more than 70")

        predicted_age = max(prediction, key=prediction.get)
        return cls(
            predicted_age=predicted_age, probabilities=AgeProbabilities(**prediction)
        )
