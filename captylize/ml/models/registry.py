from captylize.ml.manager import ModelManager
from captylize.ml.models.config import (
    ModelCategory,
    AnalysesType,
    GenerationType,
    Florence2Task,
    AgeModelName,
    EmotionModelName,
    NSFWModelName,
    VITCaptionModelName,
    Florence2ModelName,
)


from captylize.ml.models.vit_model import ViTImg2TextModel
from captylize.ml.models.caption.basic.vit_gpt2_image_captioning import (
    VitGPT2CaptionModel,
)
from captylize.ml.models.caption.advanced.florence_2 import Florence2StandardModel
from captylize.ml.models.caption.advanced.florence_2 import Florence2PromptGenModel
from captylize.ml.models.caption.advanced.florence_2 import Florence2FluxModel


def register_models(model_manager: ModelManager):
    model_manager.register_model(
        ModelCategory.ANALYSES,
        AnalysesType.AGE,
        AgeModelName.VIT_AGE_CLASSIFIER,
        "nateraw/vit-age-classifier",
        ViTImg2TextModel,
        is_default=True,
    )
    model_manager.register_model(
        ModelCategory.ANALYSES,
        AnalysesType.EMOTION,
        EmotionModelName.VIT_EMOTION_CLASSIFIER,
        "dima806/facial_emotions_image_detection",
        ViTImg2TextModel,
        is_default=True,
    )
    model_manager.register_model(
        ModelCategory.ANALYSES,
        AnalysesType.NSFW,
        NSFWModelName.VIT_NSFW_DETECTOR,
        "AdamCodd/vit-base-nsfw-detector",
        ViTImg2TextModel,
        is_default=True,
    )
    # VIT CAPTION REGISTRATION

    model_manager.register_model(
        ModelCategory.GENERATION,
        GenerationType.VIT_CAPTION,
        VITCaptionModelName.VIT_GPT2_IMAGE_CAPTIONING,
        "nlpconnect/vit-gpt2-image-captioning",
        VitGPT2CaptionModel,
        is_default=True,
    )

    # FLORENCE-2 BASED MODEL REGISTRATION

    model_manager.register_model(
        ModelCategory.GENERATION,
        GenerationType.FLORENCE2_CAPTION,
        Florence2ModelName.STANDARD_LARGE,
        "microsoft/Florence-2-large",
        Florence2StandardModel,
        is_default=True,
        available_tasks=[
            Florence2Task.CAPTION,
            Florence2Task.DETAILED_CAPTION,
            Florence2Task.MORE_DETAILED_CAPTION,
        ],
        default_task=Florence2Task.CAPTION,
    )

    model_manager.register_model(
        ModelCategory.GENERATION,
        GenerationType.FLORENCE2_CAPTION,
        Florence2ModelName.STANDARD_BASE,
        "microsoft/Florence-2-base",
        Florence2StandardModel,
        available_tasks=[
            Florence2Task.CAPTION,
            Florence2Task.DETAILED_CAPTION,
            Florence2Task.MORE_DETAILED_CAPTION,
        ],
        default_task=Florence2Task.CAPTION,
    )

    model_manager.register_model(
        ModelCategory.GENERATION,
        GenerationType.FLORENCE2_CAPTION,
        Florence2ModelName.PROMPTGEN_LARGE,
        "MiaoshouAI/Florence-2-large-PromptGen-v1.5",
        Florence2PromptGenModel,
        available_tasks=[
            Florence2Task.CAPTION,
            Florence2Task.DETAILED_CAPTION,
            Florence2Task.MORE_DETAILED_CAPTION,
            Florence2Task.GENERATE_TAGS,
            Florence2Task.MIXED_CAPTION,
        ],
        default_task=Florence2Task.CAPTION,
    )

    model_manager.register_model(
        ModelCategory.GENERATION,
        GenerationType.FLORENCE2_CAPTION,
        Florence2ModelName.PROMPTGEN_BASE,
        "MiaoshouAI/Florence-2-base-PromptGen-v1.5",
        Florence2PromptGenModel,
        available_tasks=[
            Florence2Task.CAPTION,
            Florence2Task.DETAILED_CAPTION,
            Florence2Task.MORE_DETAILED_CAPTION,
            Florence2Task.GENERATE_TAGS,
            Florence2Task.MIXED_CAPTION,
        ],
        default_task=Florence2Task.CAPTION,
    )

    model_manager.register_model(
        ModelCategory.GENERATION,
        GenerationType.FLORENCE2_CAPTION,
        Florence2ModelName.FLUX_LARGE,
        "gokaygokay/Florence-2-Flux-Large",
        Florence2FluxModel,
        available_tasks=[
            Florence2Task.DESCRIPTION,
        ],
        default_task=Florence2Task.DESCRIPTION,
    )

    model_manager.register_model(
        ModelCategory.GENERATION,
        GenerationType.FLORENCE2_CAPTION,
        Florence2ModelName.FLUX_BASE,
        "gokaygokay/Florence-2-Flux",
        Florence2FluxModel,
        available_tasks=[
            Florence2Task.DESCRIPTION,
        ],
        default_task=Florence2Task.DESCRIPTION,
    )
