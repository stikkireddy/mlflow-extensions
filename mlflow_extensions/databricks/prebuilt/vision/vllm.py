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

QWEN2_VL_2B_INSTRUCT = EzDeployConfig(
    name="qwen2_vl_2b_instruct",
    engine_proc=_ENGINE,
    engine_config=_ENGINE_CONFIG(
        model="Qwen/Qwen2-VL-2B-Instruct",
        guided_decoding_backend="outlines",
        vllm_command_flags={
            "--gpu-memory-utilization": 0.98,
            "--distributed-executor-backend": "ray",
        },
        max_num_images=5,
        # TODO FIX THIS ON NEW RELEASE OF TRANSFORMERS 0.45.0 otherwise you will get qwen2_vl not found
        # https://github.com/huggingface/transformers/issues/33401
        # https://github.com/QwenLM/Qwen2-VL?tab=readme-ov-file#quickstart
        library_overrides={
            "transformers": "git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830",
            "accelerate": "accelerate==0.31.0",
            "vllm": "vllm==0.6.1",
        },
    ),
    serving_config=ServingConfig(
        # rough estimate for the engines this includes model weights + kv cache + overhead + intermediate states
        minimum_memory_in_gb=20,
    ),
)

QWEN2_VL_7B_INSTRUCT = EzDeployConfig(
    name="qwen2_vl_7b_instruct",
    engine_proc=_ENGINE,
    engine_config=_ENGINE_CONFIG(
        model="Qwen/Qwen2-VL-7B-Instruct",
        guided_decoding_backend="outlines",
        vllm_command_flags={
            "--gpu-memory-utilization": 0.98,
            "--distributed-executor-backend": "ray",
        },
        max_num_images=5,
        # TODO FIX THIS ON NEW RELEASE OF TRANSFORMERS 0.45.0 otherwise you will get qwen2_vl not found
        # https://github.com/huggingface/transformers/issues/33401
        # https://github.com/QwenLM/Qwen2-VL?tab=readme-ov-file#quickstart
        library_overrides={
            "transformers": "git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830",
            "accelerate": "accelerate==0.31.0",
            "vllm": "vllm==0.6.1",
        },
    ),
    serving_config=ServingConfig(
        # rough estimate for the engines this includes model weights + kv cache + overhead + intermediate states
        minimum_memory_in_gb=70,
    ),
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
    QWEN2_VL_2B_INSTRUCT: EzDeployConfig = field(
        default_factory=lambda: QWEN2_VL_2B_INSTRUCT
    )
    QWEN2_VL_7B_INSTRUCT: EzDeployConfig = field(
        default_factory=lambda: QWEN2_VL_7B_INSTRUCT
    )
