from dataclasses import dataclass, field

from mlflow_extensions.databricks.prebuilt.vision.sglang import SglangVision
from mlflow_extensions.databricks.prebuilt.vision.vllm import VllmVision


@dataclass(frozen=True)
class VisionLM:
    sglang: SglangVision = field(default_factory=lambda: SglangVision())
    vllm: VllmVision = field(default_factory=lambda: VllmVision())
