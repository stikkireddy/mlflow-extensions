from mlflow_extensions.serving.databricks.lms.ez_deploy import (
    EzDeployConfig,
    ServingConfig,
)
from mlflow_extensions.serving.engines import SglangEngineProcess, SglangEngineConfig

ENGINE = SglangEngineProcess
ENGINE_CONFIG = SglangEngineConfig

llava_model_serving_configs_7b = ServingConfig(
    # rough estimate for the engines this includes model weights + kv cache + overhead + intermediate states
    minimum_memory_in_gb=30,
)

llava_model_serving_configs_34b = ServingConfig(
    # rough estimate for the engines this includes model weights + kv cache + overhead + intermediate states
    minimum_memory_in_gb=70,
)

# https://huggingface.co/liuhaotian/llava-v1.5-7b
llava_1_5_llama = EzDeployConfig(
    name="llava_1_5_llama_based",
    engine_proc=ENGINE,
    engine_config=ENGINE_CONFIG(
        model="liuhaotian/llava-v1.5-7b",
        tokenizer_path="llava-hf/llava-1.5-7b-hf",
        chat_template_builtin_name="vicuna_v1.1",
        verify_chat_template=False,  # it will use the built in chat template
    ),
    serving_config=llava_model_serving_configs_7b,
)

# https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b
llava_1_6_vicuna = EzDeployConfig(
    name="llava_1_6_7b_vicuna_based",
    engine_proc=ENGINE,
    engine_config=ENGINE_CONFIG(
        model="liuhaotian/llava-v1.6-vicuna-7b",
        tokenizer_path="llava-hf/llava-1.5-7b-hf",
        chat_template_builtin_name="vicuna_v1.1",
        verify_chat_template=False,  # it will use the built in chat template
    ),
    serving_config=llava_model_serving_configs_7b,
)

# https://huggingface.co/liuhaotian/llava-v1.6-34b
llava_1_6_yi_34b = EzDeployConfig(
    name="llava_1_6_34b_yi_based",
    engine_proc=ENGINE,
    engine_config=ENGINE_CONFIG(
        model="liuhaotian/llava-v1.6-34b",
        tokenizer_path="liuhaotian/llava-v1.6-34b-tokenizer",
    ),
    serving_config=llava_model_serving_configs_7b,
)

# https://huggingface.co/lmms-lab/llama3-llava-next-8b
llava_next_llama3_8b = EzDeployConfig(
    name="llava_next_llama3_8b_based",
    engine_proc=ENGINE,
    engine_config=ENGINE_CONFIG(
        model="lmms-lab/llama3-llava-next-8b",
        tokenizer_path="lmms-lab/llama3-llava-next-8b-tokenizer",
    ),
    serving_config=llava_model_serving_configs_7b,
)
