from dataclasses import dataclass, field

from mlflow_extensions.databricks.deploy.ez_deploy import (
    EzDeployConfig,
    ServingConfig,
)
from mlflow_extensions.serving.engines import SglangEngineProcess, SglangEngineConfig

_ENGINE = SglangEngineProcess
_ENGINE_CONFIG = SglangEngineConfig

_llava_model_serving_configs_7b = ServingConfig(
    # rough estimate for the engines this includes model weights + kv cache + overhead + intermediate states
    minimum_memory_in_gb=30,
)

_llava_model_serving_configs_34b = ServingConfig(
    # rough estimate for the engines this includes model weights + kv cache + overhead + intermediate states
    minimum_memory_in_gb=70,
)

# https://huggingface.co/liuhaotian/llava-v1.5-7b
LLAVA_1_5_LLAMA_CONFIG = EzDeployConfig(
    name="llava_1_5_llama_based",
    engine_proc=_ENGINE,
    engine_config=_ENGINE_CONFIG(
        model="liuhaotian/llava-v1.5-7b",
        tokenizer_path="llava-hf/llava-1.5-7b-hf",
        chat_template_builtin_name="vicuna_v1.1",
        verify_chat_template=False,  # it will use the built in chat template
    ),
    serving_config=_llava_model_serving_configs_7b,
)

# https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b
LLAVA_1_6_VICUNA_CONFIG = EzDeployConfig(
    name="llava_1_6_7b_vicuna_based",
    engine_proc=_ENGINE,
    engine_config=_ENGINE_CONFIG(
        model="liuhaotian/llava-v1.6-vicuna-7b",
        tokenizer_path="llava-hf/llava-1.5-7b-hf",
        chat_template_builtin_name="vicuna_v1.1",
        verify_chat_template=False,  # it will use the built in chat template
    ),
    serving_config=_llava_model_serving_configs_7b,
)

# https://huggingface.co/liuhaotian/llava-v1.6-34b
LLAVA_1_6_YI_34B_CONFIG = EzDeployConfig(
    name="llava_1_6_34b_yi_based",
    engine_proc=_ENGINE,
    engine_config=_ENGINE_CONFIG(
        model="liuhaotian/llava-v1.6-34b",
        tokenizer_path="liuhaotian/llava-v1.6-34b-tokenizer",
    ),
    serving_config=_llava_model_serving_configs_7b,
)

# https://huggingface.co/lmms-lab/llama3-llava-next-8b
LLAVA_NEXT_LLAMA3_8B_CONFIG = EzDeployConfig(
    name="llava_next_llama3_8b_based",
    engine_proc=_ENGINE,
    engine_config=_ENGINE_CONFIG(
        model="lmms-lab/llama3-llava-next-8b",
        tokenizer_path="lmms-lab/llama3-llava-next-8b-tokenizer",
    ),
    serving_config=_llava_model_serving_configs_7b,
)


@dataclass(frozen=True)
class SglangVision:
    LLAVA_1_5_LLAMA: EzDeployConfig = field(
        default_factory=lambda: LLAVA_1_5_LLAMA_CONFIG
    )
    LLAVA_1_6_VICUNA: EzDeployConfig = field(
        default_factory=lambda: LLAVA_1_6_VICUNA_CONFIG
    )
    LLAVA_1_6_YI_34B: EzDeployConfig = field(
        default_factory=lambda: LLAVA_1_6_YI_34B_CONFIG
    )
    LLAVA_NEXT_LLAMA3_8B: EzDeployConfig = field(
        default_factory=lambda: LLAVA_NEXT_LLAMA3_8B_CONFIG
    )
