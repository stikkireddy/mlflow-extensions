from dataclasses import dataclass, field

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployConfig, ServingConfig
from mlflow_extensions.serving.engines import VLLMEngineConfig, VLLMEngineProcess

_ENGINE = VLLMEngineProcess
_ENGINE_CONFIG = VLLMEngineConfig

# https://huggingface.co/fixie-ai/ultravox-v0_4
FIXIE_ULTRA_VOX_0_4_64K_CONFIG = EzDeployConfig(
    name="fixie_utravox_0_4_llama_3_1_whisper_medium",
    engine_proc=_ENGINE,
    engine_config=_ENGINE_CONFIG(
        model="fixie-ai/ultravox-v0_4",
        trust_remote_code=True,
        guided_decoding_backend="outlines",
        max_model_len=64000,
    ),
    serving_config=ServingConfig(
        # rough estimate for the engines this includes model weights + kv cache + overhead + intermediate states
        minimum_memory_in_gb=60,
    ),
)

# https://huggingface.co/fixie-ai/ultravox-v0_4
FIXIE_ULTRA_VOX_0_4_128K_CONFIG = EzDeployConfig(
    name="fixie_utravox_0_4_llama_3_1_whisper_medium",
    engine_proc=_ENGINE,
    engine_config=_ENGINE_CONFIG(
        model="fixie-ai/ultravox-v0_4",
        trust_remote_code=True,
        guided_decoding_backend="outlines",
    ),
    serving_config=ServingConfig(
        # rough estimate for the engines this includes model weights + kv cache + overhead + intermediate states
        minimum_memory_in_gb=110,
    ),
)


@dataclass(frozen=True)
class VllmAudio:
    FIXIE_ULTRA_VOX_0_4_64K_CONFIG: EzDeployConfig = field(
        default_factory=lambda: FIXIE_ULTRA_VOX_0_4_64K_CONFIG
    )
    FIXIE_ULTRA_VOX_0_4_128K_CONFIG: EzDeployConfig = field(
        default_factory=lambda: FIXIE_ULTRA_VOX_0_4_128K_CONFIG
    )
