from dataclasses import dataclass, field

from mlflow_extensions.databricks.prebuilt.text.sglang import SglangText
from mlflow_extensions.databricks.prebuilt.text.vllm import VllmText


@dataclass(frozen=True)
class TextLM:
    sglang: SglangText = field(default_factory=lambda: SglangText())
    vllm: VllmText = field(default_factory=lambda: VllmText())
