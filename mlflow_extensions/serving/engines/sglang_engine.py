import sys
from dataclasses import field, dataclass
from pathlib import Path
from typing import List, Dict, Optional, Union

from mlflow.pyfunc import PythonModelContext

from mlflow_extensions.serving.engines import gpu_utils
from mlflow_extensions.serving.engines.base import EngineConfig, debug_msg, EngineProcess, Command
from mlflow_extensions.serving.engines.huggingface_utils import snapshot_download_local, ensure_chat_template


@dataclass(frozen=True, kw_only=True)
class SglangEngineConfig(EngineConfig):
    entrypoint_module: str = field(default="sglang.launch_server")
    sglang_command_flags: Dict[str, Optional[str]] = field(default_factory=dict)
    trust_remote_code: bool = field(default=False)  # --trust-remote-code
    context_length: Optional[int] = field(default=None)  # --context-length
    served_model_alias: Optional[str] = field(default=None)  # --served-model-name
    quantization: Optional[str] = field(default=None)  # --quantization
    # generic
    model_artifact_key: str = field(default="model")
    verify_chat_template: bool = field(default=True)
    tokenizer_config_file: str = field(default="tokenizer_config.json")
    chat_template_key: str = field(default="chat_template")

    def _to_sglang_command(self, context: PythonModelContext = None) -> List[str]:
        local_model_path = None
        if context is not None:
            local_model_path = context.artifacts.get(self.model_artifact_key)
        flags = []

        # add tensor parallel size flag if we have GPUs
        gpu_count = gpu_utils.get_gpu_count()
        if gpu_count >= 1:
            flags.append("--tensor-parallel-size")
            flags.append(str(gpu_count))

        skip_flags = ["--model-path",
                      "--context-length",
                      "--trust-remote-code",
                      "--served-model-name",
                      "--quantization"]
        for k, v in self.sglang_command_flags.items():
            if k in skip_flags:
                debug_msg(f"Skipping flag {k} use the built in argument")
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

    def _to_run_command(self, context: PythonModelContext = None) -> Union[List[str], Command]:
        return self._to_sglang_command(context=context)

    def engine_pip_reqs(self, *,
                        sglang_version: str = "0.2.13",
                        flashinfer_extra_index_url: str = "https://flashinfer.ai/whl/cu121/torch2.4/",
                        flashinfer_version: str = "0.1.6") -> List[str]:
        default_installs = [f"sglang[all]=={sglang_version}"]
        if flashinfer_extra_index_url is not None:
            default_installs.append(f"--extra-index-url={flashinfer_extra_index_url}")
        default_installs.append(f"flashinfer=={flashinfer_version}")
        return default_installs

    def _setup_snapshot(self, local_dir: str = "/root/models"):
        return snapshot_download_local(repo_id=self.model, local_dir=local_dir)

    def _setup_artifacts(self, local_dir: str = "/root/models"):
        local_path = self._setup_snapshot(local_dir)
        return {self.model_artifact_key: local_path}

    def _verify_chat_template(self, artifacts: Dict[str, str]):
        model_dir_path = Path(artifacts[self.model_artifact_key])
        tokenizer_config_file = model_dir_path / self.tokenizer_config_file
        ensure_chat_template(tokenizer_file=str(tokenizer_config_file),
                             chat_template_key=self.chat_template_key)

    def setup_artifacts(self, local_dir: str = "/root/models"):
        artifacts = self._setup_artifacts(local_dir)
        if self.verify_chat_template is True:
            self._verify_chat_template(artifacts)
        return artifacts

    def supported_models(self) -> List[str]:
        return []


class SglangEngineProcess(EngineProcess):

    @property
    def engine_name(self) -> str:
        return "sglang-engine"

    def health_check(self) -> bool:
        try:
            resp = self.server_http_client.get("/health")
            return resp.status_code == 200
        except Exception as e:
            debug_msg(f"Health check failed with error {e}; server may not be up yet or crashed;")
            return False
