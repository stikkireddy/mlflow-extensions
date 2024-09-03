from dataclasses import dataclass, field

from mlflow_extensions.databricks.prebuilt.text import TextLM
from mlflow_extensions.databricks.prebuilt.vision import VisionLM


@dataclass(frozen=True)
class Prebuilt:
    text: TextLM = field(default_factory=lambda: TextLM())
    vision: VisionLM = field(default_factory=lambda: VisionLM())


prebuilt = Prebuilt()
