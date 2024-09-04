from dataclasses import dataclass, field

from mlflow_extensions.databricks.prebuilt.audio.vllm import VllmAudio


@dataclass(frozen=True)
class AudioLM:

    vllm: VllmAudio = field(default_factory=lambda: VllmAudio())
