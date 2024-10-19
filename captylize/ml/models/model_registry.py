from typing import Literal


MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "analyses": {
        "age": {
            "vit_age_classifier": "nateraw/vit-age-classifier",
        },
        "emotion": {
            "vit_emotion_classifier": "dima806/vit-emotion-classifier",
        },
        "nsfw": {
            "vit_nsfw_detector": "AdamCodd/vit-base-nsfw-detector",
        },
    }
}

DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "analyses": {
        "age": "vit_age_classifier",
        "emotion": "vit_emotion_classifier",
        "nsfw": "vit_nsfw_detector",
    }
}

ModelType = Literal["age", "emotion", "nsfw"]
