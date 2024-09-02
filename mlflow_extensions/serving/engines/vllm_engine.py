import sys
from dataclasses import field, dataclass
from pathlib import Path
from typing import List, Dict, Optional, Union

from mlflow.pyfunc import PythonModelContext

from mlflow_extensions.serving.engines import gpu_utils
from mlflow_extensions.serving.engines.base import (
    EngineConfig,
    debug_msg,
    EngineProcess,
    Command,
)
from mlflow_extensions.serving.engines.huggingface_utils import (
    snapshot_download_local,
    ensure_chat_template,
)


@dataclass(frozen=True, kw_only=True)
class VLLMEngineConfig(EngineConfig):
    entrypoint_module: str = field(default="vllm.entrypoints.openai.api_server")
    enable_experimental_chunked_prefill: bool = field(default=False)
    max_num_batched_tokens: int = field(
        default=512
    )  # 512 is based on A100 ITL for llama model
    enable_prefix_caching: bool = field(default=False)
    vllm_command_flags: Dict[str, Optional[str]] = field(default_factory=dict)
    trust_remote_code: bool = field(default=False)
    max_model_len: Optional[int] = field(default=None)
    served_model_alias: Optional[str] = field(default=None)
    guided_decoding_backend: Optional[str] = field(default="outlines")
    tokenizer: Optional[str] = field(default=None)  # --tokenizer

    # general keys
    model_artifact_key: str = field(default="model")
    verify_chat_template: bool = field(default=True)
    tokenizer_artifact_key: str = field(default="tokenizer")
    tokenizer_config_file: str = field(default="tokenizer_config.json")
    chat_template_key: str = field(default="chat_template")

    def _to_vllm_command(self, context: PythonModelContext = None) -> List[str]:
        local_model_path = None
        tokenizer_path = None
        if context is not None:
            local_model_path = context.artifacts.get(self.model_artifact_key)
            tokenizer_path = context.artifacts.get(self.tokenizer_artifact_key)
        flags = []
        skip_flags = [
            "--enable-chunked-prefill",
            "--model",
            "--tokenizer",
            "--max-num-batched-tokens",
            "--enable-prefix-caching",
            "--max-model-len",
            "--trust-remote-code",
            "--served-model-name",
            "--guided-decoding-backend",
        ]

        # add tensor parallel size flag if we have GPUs
        gpu_count = gpu_utils.get_gpu_count()
        if gpu_count >= 1:
            flags.append("--tensor-parallel-size")
            flags.append(str(gpu_count))

        for k, v in self.vllm_command_flags.items():
            if k in skip_flags:
                debug_msg(f"Skipping flag {k} use the built in argument")
                continue
            flags.append(k)
            if v is not None:
                flags.append(v)
        if self.enable_experimental_chunked_prefill is True:
            flags.append("--enable-chunked-prefill")
            flags.append("--max-num-batched-tokens")
            flags.append(str(self.max_num_batched_tokens))
        if self.enable_prefix_caching is True:
            flags.append("--enable-prefix-caching")
        if self.trust_remote_code is True:
            flags.append("--trust-remote-code")
        if self.max_model_len is not None:
            flags.append("--max-model-len")
            flags.append(str(self.max_model_len))
        if self.guided_decoding_backend is not None:
            flags.append("--guided-decoding-backend")
            if self.guided_decoding_backend not in ["outlines", "lm-format-enforcer"]:
                raise ValueError(
                    f"Invalid guided decoding backend {self.guided_decoding_backend}"
                )
            flags.append(self.guided_decoding_backend)
        if self.tokenizer is not None:
            flags.append("--tokenizer")
            flags.append(tokenizer_path or self.tokenizer)

        return [
            sys.executable,
            "-m",
            self.entrypoint_module,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--model",
            self.model if local_model_path is None else local_model_path,
            "--served-model-name",
            self.model if self.served_model_alias is None else self.served_model_alias,
            *flags,
        ]

    def _to_run_command(
        self, context: PythonModelContext = None
    ) -> Union[List[str], Command]:
        return self._to_vllm_command(context=context)

    def engine_pip_reqs(
        self,
        *,
        vllm_version: str = "0.5.5",
        lm_format_enforcer_version: str = "0.10.6",
        outlines_version: str = "0.0.46",
    ) -> List[str]:
        default_installs = [f"vllm=={vllm_version}"]
        if self.guided_decoding_backend == "lm-format-enforcer":
            default_installs.append(f"lm-format-enforcer=={lm_format_enforcer_version}")
        if self.guided_decoding_backend == "outlines":
            default_installs.append(f"outlines=={outlines_version}")
        return default_installs

    def _setup_snapshot(self, local_dir: str = "/root/models"):
        return snapshot_download_local(repo_id=self.model, local_dir=local_dir)

    def _setup_artifacts(self, local_dir: str = "/root/models"):
        local_path = self._setup_snapshot(local_dir)
        # tokenizer path
        artifacts = {self.model_artifact_key: local_path}
        if self.tokenizer is not None and self.model != self.tokenizer:
            tokenizer_local_path = snapshot_download_local(
                repo_id=self.tokenizer, local_dir=local_dir, tokenizer_only=True
            )
            artifacts[self.tokenizer_artifact_key] = tokenizer_local_path
        return artifacts

    def _verify_chat_template(self, artifacts: Dict[str, str]):
        if self.tokenizer is None:
            model_dir_path = Path(artifacts[self.model_artifact_key])
        else:
            model_dir_path = Path(artifacts[self.tokenizer_artifact_key])
        tokenizer_config_file = model_dir_path / self.tokenizer_config_file
        ensure_chat_template(
            tokenizer_file=str(tokenizer_config_file),
            chat_template_key=self.chat_template_key,
        )

    def setup_artifacts(self, local_dir: str = "/root/models"):
        artifacts = self._setup_artifacts(local_dir)
        if self.verify_chat_template is True:
            self._verify_chat_template(artifacts)
        return artifacts

    def supported_model_architectures(self) -> List[str]:
        return [
            "AquilaForCausalLM",
            "ArcticForCausalLM",
            "BaiChuanForCausalLM",
            "BloomForCausalLM",
            "ChatGLMModel",
            "CohereForCausalLM",
            "DbrxForCausalLM",
            "DeciLMForCausalLM",
            "ExaoneForCausalLM",
            "FalconForCausalLM",
            "GemmaForCausalLM",
            "Gemma2ForCausalLM",
            "GPT2LMHeadModel",
            "GPTBigCodeForCausalLM",
            "GPTJForCausalLM",
            "GPTNeoXForCausalLM",
            "InternLMForCausalLM",
            "InternLM2ForCausalLM",
            "JAISLMHeadModel",
            "JambaForCausalLM",
            "LlamaForCausalLM",
            "MiniCPMForCausalLM",
            "MistralForCausalLM",
            "MixtralForCausalLM",
            "MPTForCausalLM",
            "NemotronForCausalLM",
            "OLMoForCausalLM",
            "OPTForCausalLM",
            "OrionForCausalLM",
            "PhiForCausalLM",
            "Phi3ForCausalLM",
            "Phi3SmallForCausalLM",
            "PhiMoEForCausalLM",
            "PersimmonForCausalLM",
            "QWenLMHeadModel",
            "Qwen2ForCausalLM",
            "Qwen2MoeForCausalLM",
            "StableLmForCausalLM",
            "Starcoder2ForCausalLM",
            "XverseForCausalLM",
            "Blip2ForConditionalGeneration",
            "ChameleonForConditionalGeneration",
            "FuyuForCausalLM",
            "InternVLChatModel",
            "LlavaForConditionalGeneration",
            "LlavaNextForConditionalGeneration",
            "PaliGemmaForConditionalGeneration",
            "Phi3VForCausalLM",
            "MiniCPMV",
            "UltravoxModel",
        ]


class VLLMEngineProcess(EngineProcess):

    @property
    def engine_name(self) -> str:
        return "vllm-engine"

    def health_check(self) -> bool:
        try:
            resp = self.server_http_client.get("/health")
            return resp.status_code == 200
        except Exception as e:
            debug_msg(
                f"Health check failed with error {e}; server may not be up yet or crashed;"
            )
            return False
