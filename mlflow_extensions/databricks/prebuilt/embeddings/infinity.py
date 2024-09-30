from dataclasses import dataclass, field

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployConfig, ServingConfig
from mlflow_extensions.serving.engines import (
    InfinityEngineConfig,
    InfinityEngineProcess,
)

_ENGINE = InfinityEngineProcess
_ENGINE_CONFIG = InfinityEngineConfig

# https://huggingface.co/BAAI/bge-large-en-v1.5
BGE_LARGE_EN_1_5_CONFIG = EzDeployConfig(
    name="bgelarge_en_1_5",
    engine_proc=_ENGINE,
    engine_config=_ENGINE_CONFIG(
        model="BAAI/bge-large-en-v1.5", torch_compile=True, batch_size=512
    ),
    serving_config=ServingConfig(
        minimum_memory_in_gb=10,
    ),
)


@dataclass(frozen=True)
class InfinityEmbeddings:
    BGE_LARGE_EN_1_5_CONFIG: EzDeployConfig = field(
        default_factory=lambda: BGE_LARGE_EN_1_5_CONFIG
    )
