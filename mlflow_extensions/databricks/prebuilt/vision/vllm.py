from dataclasses import dataclass, field
from typing import Optional

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployConfig, ServingConfig
from mlflow_extensions.serving.engines import VLLMEngineConfig, VLLMEngineProcess

_ENGINE = VLLMEngineProcess
_ENGINE_CONFIG = VLLMEngineConfig


def phi_3_5_vision_instruct(
    ctx_name: str,
    context_length: Optional[int],
    min_memory_gb: int,
    max_num_images: int = 1,
) -> EzDeployConfig:
    model_cfg = {
        "max_model_len": context_length,
    }
    return EzDeployConfig(
        name=f"phi_3_5_vision_instruct_{ctx_name}",
        engine_proc=_ENGINE,
        engine_config=_ENGINE_CONFIG(
            model="microsoft/Phi-3.5-vision-instruct",
            trust_remote_code=True,
            guided_decoding_backend="outlines",
            max_num_images=max_num_images,
            vllm_command_flags={
                "--gpu-memory-utilization": 0.98,
                "--enforce-eager": None,
            },
            **model_cfg,
        ),
        serving_config=ServingConfig(
            # rough estimate for the engines this includes model weights + kv cache + overhead + intermediate states
            minimum_memory_in_gb=min_memory_gb,
        ),
    )


# https://huggingface.co/microsoft/Phi-3.5-vision-instruct
PHI_3_5_VISION_INSTRUCT_4K_CONFIG = phi_3_5_vision_instruct(
    "4k", 4096, 12, max_num_images=1
)

# https://huggingface.co/microsoft/Phi-3.5-vision-instruct
PHI_3_5_VISION_INSTRUCT_8K_CONFIG = phi_3_5_vision_instruct(
    "8k", 8192, 15, max_num_images=2
)

# https://huggingface.co/microsoft/Phi-3.5-vision-instruct
PHI_3_5_VISION_INSTRUCT_12K_CONFIG = phi_3_5_vision_instruct(
    "12k", 12000, 20, max_num_images=4
)

# https://huggingface.co/microsoft/Phi-3.5-vision-instruct
PHI_3_5_VISION_INSTRUCT_32K_CONFIG = phi_3_5_vision_instruct(
    "32k", 32000, 30, max_num_images=4
)

# https://huggingface.co/microsoft/Phi-3.5-vision-instruct
PHI_3_5_VISION_INSTRUCT_64K_CONFIG = phi_3_5_vision_instruct(
    "64k", 64000, 50, max_num_images=8
)

# https://huggingface.co/microsoft/Phi-3.5-vision-instruct
PHI_3_5_VISION_INSTRUCT_128K_CONFIG = phi_3_5_vision_instruct(
    "128k", None, 110, max_num_images=8
)


@dataclass(frozen=True)
class VllmVision:
    PHI_3_5_VISION_INSTRUCT_4K: EzDeployConfig = field(
        default_factory=lambda: PHI_3_5_VISION_INSTRUCT_4K_CONFIG
    )
    PHI_3_5_VISION_INSTRUCT_8K: EzDeployConfig = field(
        default_factory=lambda: PHI_3_5_VISION_INSTRUCT_8K_CONFIG
    )
    PHI_3_5_VISION_INSTRUCT_12K: EzDeployConfig = field(
        default_factory=lambda: PHI_3_5_VISION_INSTRUCT_12K_CONFIG
    )
    PHI_3_5_VISION_INSTRUCT_32K: EzDeployConfig = field(
        default_factory=lambda: PHI_3_5_VISION_INSTRUCT_32K_CONFIG
    )
    PHI_3_5_VISION_INSTRUCT_64K: EzDeployConfig = field(
        default_factory=lambda: PHI_3_5_VISION_INSTRUCT_64K_CONFIG
    )
    PHI_3_5_VISION_INSTRUCT_128K: EzDeployConfig = field(
        default_factory=lambda: PHI_3_5_VISION_INSTRUCT_128K_CONFIG
    )
