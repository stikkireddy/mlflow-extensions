from dataclasses import dataclass, field

from mlflow_extensions.databricks.deploy.ez_deploy import (
    EzDeployConfig,
    ServingConfig,
)
from mlflow_extensions.serving.engines import VLLMEngineProcess, VLLMEngineConfig

_ENGINE = VLLMEngineProcess
_ENGINE_CONFIG = VLLMEngineConfig

# https://huggingface.co/numind/NuExtract
NUEXTRACT_CONFIG = EzDeployConfig(
    name="nuextract_small",
    engine_proc=_ENGINE,
    engine_config=_ENGINE_CONFIG(
        model="numind/NuExtract",
        trust_remote_code=True,
        guided_decoding_backend="outlines",
    ),
    serving_config=ServingConfig(
        # rough estimate for the engines this includes model weights + kv cache + overhead + intermediate states
        minimum_memory_in_gb=20,
    ),
)

# https://huggingface.co/numind/NuExtract-tiny
NUEXTRACT_TINY_CONFIG = EzDeployConfig(
    name="nuextract_tiny",
    engine_proc=_ENGINE,
    engine_config=_ENGINE_CONFIG(
        model="numind/NuExtract-tiny",
        trust_remote_code=True,
        guided_decoding_backend="outlines",
    ),
    serving_config=ServingConfig(
        # rough estimate for the engines this includes model weights + kv cache + overhead + intermediate states
        minimum_memory_in_gb=10,
    ),
)


# https://huggingface.co/microsoft/Phi-3.5-MoE-instruct
# PHI_3_5_MOE_INSTRUCT_8K = EzDeployConfig(
#     name="phi_3_5_moe_instruct_8k",
#     engine_proc=_ENGINE,
#     engine_config=_ENGINE_CONFIG(
#         model="microsoft/Phi-3.5-MoE-instruct",
#         trust_remote_code=True,
#         max_model_len=8192,
#         guided_decoding_backend="outlines"
#     ),
#     serving_config=ServingConfig(
#         # rough estimate for the engines this includes model weights + kv cache + overhead + intermediate states
#         minimum_memory_in_gb=50,
#     ),
# )


@dataclass(frozen=True)
class VllmText:
    NUEXTRACT: EzDeployConfig = field(default_factory=lambda: NUEXTRACT_CONFIG)
    NUEXTRACT_TINY: EzDeployConfig = field(
        default_factory=lambda: NUEXTRACT_TINY_CONFIG
    )
    # PHI_3_5_MOE_INSTRUCT_8K = PHI_3_5_MOE_INSTRUCT_8K
