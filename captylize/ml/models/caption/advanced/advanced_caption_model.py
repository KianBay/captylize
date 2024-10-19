from typing import Optional, ClassVar
from PIL import Image
from captylize.ml.models.img_to_text_model import Img2TextModel


class AdvancedCaptionModel(Img2TextModel[str]):
    available_tasks: ClassVar[list[str]] = []

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

        raise NotImplementedError

    @classmethod
    def get_available_tasks(cls) -> list[str]:
        return cls.available_tasks
