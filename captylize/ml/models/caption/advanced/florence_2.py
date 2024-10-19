from typing import List, Optional, ClassVar
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

from captylize.logger import get_logger
from captylize.ml.models.caption.advanced.base import AdvancedCaptionModel

logger = get_logger(__name__)


class Florence2Model(AdvancedCaptionModel):
    available_tasks: ClassVar[List[str]] = []
    default_task: ClassVar[str] = ""

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
        self.processor = None

    def _load(self) -> None:
        logger.info(f"Loading model {self.model_name} ({self.model_location})")
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_location,
                cache_dir=self.cache_dir,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_location, cache_dir=self.cache_dir, trust_remote_code=True
        )

    def _unload(self) -> None:
        self.model = None
        self.processor = None

    def _predict(
        self,
        image: Image.Image,
        task: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> str:
        if task and task not in self.available_tasks:
            raise ValueError(
                f"Invalid task. Available tasks are: {', '.join(self.available_tasks)}"
            )
        if not self.model or not self.processor:
            logger.debug("Model or processor not loaded, loading...")
            self._load()

        task_prompt = f"<{task.upper()}>" if task else f"<{self.default_task.upper()}>"
        if prompt:
            full_prompt = task_prompt + prompt
        else:
            full_prompt = task_prompt
        logger.info(
            f"Generating {task if task else self.default_task} using prompt: {prompt if prompt else '<None>'}"
        )
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        inputs = self.processor(text=full_prompt, images=image, return_tensors="pt").to(
            self.device, torch_dtype
        )

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        logger.info(f"Generated text: {generated_text}")
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )
        logger.info(f"Parsed answer: {parsed_answer}")

        return parsed_answer[task_prompt]


class Florence2StandardModel(Florence2Model):
    available_tasks: ClassVar[List[str]] = [
        "caption",
        "detailed_caption",
        "more_detailed_caption",
    ]
    default_task: ClassVar[str] = "caption"


class Florence2PromptGenModel(Florence2Model):
    available_tasks: ClassVar[List[str]] = [
        "caption",
        "detailed_caption",
        "more_detailed_caption",
        "generate_tags",
        "mixed_caption",
    ]
    default_task: ClassVar[str] = "caption"


class Florence2FluxModel(Florence2Model):
    available_tasks: ClassVar[List[str]] = ["description"]
    default_task: ClassVar[str] = "description"
