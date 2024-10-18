from fastapi import Query
from captylize.ml.manager import model_manager
from captylize.ml.models.ml_model import Img2TextModel


async def get_age_model(
    model_name: str = Query(None, description="The name of the model to use"),
) -> Img2TextModel[dict[str, float]]:
    return model_manager.get_age_model(model_name)
