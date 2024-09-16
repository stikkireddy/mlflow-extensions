import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from mlflow.pyfunc import PythonModelContext

from mlflow_extensions.log import Logger, get_logger
from mlflow_extensions.serving.engines import gpu_utils
from mlflow_extensions.serving.engines.base import Command, EngineConfig, EngineProcess
from mlflow_extensions.serving.engines.huggingface_utils import (
    ensure_chat_template,
    snapshot_download_local,
)
from mlflow_extensions.testing.helper import kill_processes_containing

LOGGER: Logger = get_logger()


@dataclass(frozen=True, kw_only=True)
class SglangEngineConfig(EngineConfig):
    entrypoint_module: str = field(default="sglang.launch_server")
    sglang_command_flags: Dict[str, Optional[str]] = field(default_factory=dict)
    trust_remote_code: bool = field(default=False)  # --trust-remote-code
    context_length: Optional[int] = field(default=None)  # --context-length
    served_model_alias: Optional[str] = field(default=None)  # --served-model-name
    quantization: Optional[str] = field(default=None)  # --quantization
    tokenizer_path: Optional[str] = field(default=None)  # --tokenizer-path
    # generic
    model_artifact_key: str = field(default="model")
    tokenizer_artifact_key: str = field(default="tokenizer")
    verify_chat_template: bool = field(default=True)
    tokenizer_config_file: str = field(default="tokenizer_config.json")
    chat_template_key: str = field(default="chat_template")
    chat_template_builtin_name: str = field(default=None)
    chat_template_json: dict = field(default=None)
    chat_template_file_name: str = field(default="sglang_custom_chat_template.json")

    def _to_sglang_command(self, context: PythonModelContext = None) -> List[str]:
        local_model_path = None
        tokenizer_model_path = None
        if context is not None:
            local_model_path = context.artifacts.get(self.model_artifact_key)
            tokenizer_model_path = context.artifacts.get(self.tokenizer_artifact_key)
        flags = []

        if self.tokenizer_path is not None:
            flags.append("--tokenizer-path")
            flags.append(
                tokenizer_model_path
                if tokenizer_model_path is not None
                else self.tokenizer_path
            )

        # add tensor parallel size flag if we have GPUs
        gpu_count = gpu_utils.get_gpu_count()
        if gpu_count >= 1:
            flags.append("--tensor-parallel-size")
            flags.append(str(gpu_count))

        skip_flags = [
            "--model-path",
            "--context-length",
            "--trust-remote-code",
            "--served-model-name",
            "--tokenizer-path",
            "--quantization",
            "--chat-template",
        ]
        for k, v in self.sglang_command_flags.items():
            if k in skip_flags:
                LOGGER.info(f"Skipping flag {k} use the built in argument")
                continue
            flags.append(k)
            if v is not None:
                flags.append(v)
        if self.trust_remote_code is True:
            flags.append("--trust-remote-code")
        if self.context_length is not None:
            flags.append("--context-length")
            flags.append(str(self.context_length))
        if self.quantization is not None:
            flags.append("--quantization")
            # only valid quantization methods supported
            if self.quantization not in [
                "awq",
                "fp8",
                "gptq",
                "marlin",
                "gptq_marlin",
                "awq_marlin",
                "squeezellm",
                "bitsandbytes",
            ]:
                raise ValueError(f"Invalid quantization {self.quantization}")
            flags.append(self.quantization)
        if self.chat_template_json or self.chat_template_builtin_name:
            flags.append("--chat-template")
            if self.chat_template_builtin_name:
                flags.append(self.chat_template_builtin_name)
            else:
                with open(self.chat_template_file_name, "w") as f:
                    f.write(json.dumps(self.chat_template_json))
                flags.append(self.chat_template_file_name)

        return [
            sys.executable,
            "-m",
            self.entrypoint_module,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--model-path",
            self.model if local_model_path is None else local_model_path,
            "--served-model-name",
            self.model if self.served_model_alias is None else self.served_model_alias,
            *flags,
        ]

    def _to_run_command(
        self, context: PythonModelContext = None
    ) -> Union[List[str], Command]:
        return self._to_sglang_command(context=context)

    def engine_pip_reqs(
        self,
        *,
        sglang_version: str = "0.3.0",
        flashinfer_extra_index_url: str = "https://flashinfer.ai/whl/cu121/torch2.4/",
        flashinfer_version: str = "0.1.6",
    ) -> Dict[str, str]:
        default_installs = {
            "sglang": f"sglang[all]=={sglang_version}",
        }
        if flashinfer_extra_index_url is not None:
            default_installs["extra_index_url"] = (
                f"--extra-index-url={flashinfer_extra_index_url}"
            )
        default_installs["flashinfer"] = f"flashinfer=={flashinfer_version}"
        return default_installs

    def _setup_snapshot(self, local_dir: str = "/root/models"):
        return snapshot_download_local(repo_id=self.model, local_dir=local_dir)

    def _setup_artifacts(self, local_dir: str = "/root/models"):
        local_path = self._setup_snapshot(local_dir)
        # tokenizer path
        artifacts = {self.model_artifact_key: local_path}
        if self.tokenizer_path is not None and self.model != self.tokenizer_path:
            tokenizer_local_path = snapshot_download_local(
                repo_id=self.tokenizer_path, local_dir=local_dir, tokenizer_only=True
            )
            artifacts[self.tokenizer_artifact_key] = tokenizer_local_path
        return artifacts

    def _verify_chat_template(self, artifacts: Dict[str, str]):
        if self.tokenizer_path is None:
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
            "MiniCPMForCausalLM",
            "StableLmForCausalLM",
            "LlavaVidForCausalLM",
            "Qwen2MoeForCausalLM",
            "Qwen2ForCausalLM",
            "GPTBigCodeForCausalLM",
            "ChatGLMForCausalLM",
            "ChatGLMModel",
            "DbrxForCausalLM",
            "LlavaLlamaForCausalLM",
            "LlavaQwenForCausalLM",
            "LlavaMistralForCausalLM",
            "InternLM2ForCausalLM",
            "QuantMixtralForCausalLM",
            "DeepseekV2ForCausalLM",
            "CohereForCausalLM",
            "MixtralForCausalLM",
            "YiVLForCausalLM",
            "LlamaEmbeddingModel",
            "MistralModel",
            "LlamaForClassification",
            "QWenLMHeadModel",
            "Gemma2ForCausalLM",
            "DeepseekForCausalLM",
            "LlamaForCausalLM",
            "GemmaForCausalLM",
            "MistralForCausalLM",
            "Grok1ModelForCausalLM",
        ]


class SglangEngineProcess(EngineProcess):

    @property
    def engine_name(self) -> str:
        return "sglang-engine"

    def health_check(self) -> bool:
        try:
            resp = self.server_http_client.get("/health")
            return resp.status_code == 200
        except Exception as e:
            LOGGER.info(
                f"Health check failed with error {e}; server may not be up yet or crashed;"
            )
            return False

    def cleanup(self) -> None:
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass

        try:
            kill_processes_containing("sglang")
            # sglang uses vllm under the hood and may spawn vllm proc
            kill_processes_containing("vllm")
            kill_processes_containing("ray")
            kill_processes_containing("from multiprocessing")
        except Exception:
            pass
