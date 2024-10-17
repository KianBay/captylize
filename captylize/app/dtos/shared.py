from pydantic import BaseModel, HttpUrl, model_validator, ValidationError
from typing import Optional


class ImageRequest(BaseModel):
    image_url: Optional[HttpUrl]
    image_file: Optional[bytes]

    @model_validator(mode="before")
    def validate_image_input(cls, values):
        if not values.get("image_url") and not values.get("image_file"):
            raise ValidationError("Either image_url or image_file must be provided.")
        return values
