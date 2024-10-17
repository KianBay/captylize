from typing import Literal
from pydantic import BaseModel, Field

AgeRange = Literal[
    "0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"
]


class AgeResponse(BaseModel):
    predicted_age: AgeRange
    probabilities: dict[str, float]

    @classmethod
    def from_prediction(cls, prediction: dict[str, float]):
        predicted_age = max(prediction, key=prediction.get)
        return cls(predicted_age=predicted_age, probabilities=prediction)
