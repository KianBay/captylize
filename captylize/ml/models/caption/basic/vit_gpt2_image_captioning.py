from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

from captylize.ml.models.caption.basic.base import BasicCaptionModel

from captylize.logger import get_logger

logger = get_logger(__name__)


class VitGPT2CaptionModel(BasicCaptionModel):
    def __init__(
        self,
        model_name: str,
        model_location: str,
        cache_dir: str,
        device: str,
        use_safetensors: bool = True,
    ):
        super().__init__(model_name, model_location, cache_dir, device, use_safetensors)
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None

    def _load(self) -> None:
        logger.info(f"Loading model {self.model_name} ({self.model_location})")
        self.model = VisionEncoderDecoderModel.from_pretrained(
            self.model_location, cache_dir=self.cache_dir
        )
        self.feature_extractor = ViTImageProcessor.from_pretrained(
            self.model_location, cache_dir=self.cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_location, cache_dir=self.cache_dir
        )

        self.model.to(self.device)

    def _unload(self) -> None:
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None

    def _predict(self, image: Image.Image) -> str:
        if not self.model or not self.feature_extractor or not self.tokenizer:
            self._load()

        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        pixel_values = self.feature_extractor(
            images=[image], return_tensors="pt"
        ).pixel_values
        pixel_values = pixel_values.to(self.device)

        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

        output_ids = self.model.generate(pixel_values, **gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return preds[0].strip()
