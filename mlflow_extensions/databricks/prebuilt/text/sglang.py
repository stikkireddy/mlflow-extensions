from dataclasses import dataclass, field

from mlflow_extensions.databricks.deploy.ez_deploy import (
    EzDeployConfig,
    ServingConfig,
)
from mlflow_extensions.serving.engines import SglangEngineProcess, SglangEngineConfig

_ENGINE = SglangEngineProcess
_ENGINE_CONFIG = SglangEngineConfig

# Requires HF token
# https://huggingface.co/google/gemma-2-9b-it
GEMMA_2_9B_IT_CONFIG = EzDeployConfig(
    name=f"gemma_2_9b_it",
    engine_proc=_ENGINE,
    engine_config=_ENGINE_CONFIG(model="google/gemma-2-9b-it"),
    serving_config=ServingConfig(
        # rough estimate for the engines this includes model weights + kv cache + overhead + intermediate states
        minimum_memory_in_gb=30,
    ),
)


@dataclass(frozen=True)
class SglangText:
    GEMMA_2_9B_IT: EzDeployConfig = field(default_factory=lambda: GEMMA_2_9B_IT_CONFIG)
