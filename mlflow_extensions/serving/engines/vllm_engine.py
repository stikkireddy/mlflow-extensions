import sys
from dataclasses import field, dataclass
from typing import List, Dict, Optional

from mlflow_extensions.serving.engines.base import EngineConfig, debug_msg, EngineProcess


@dataclass(frozen=True, kw_only=True)
class VLLMEngineConfig(EngineConfig):
    entrypoint_module: str = field(default="vllm.entrypoints.openai.api_server")
    enable_experimental_chunked_prefill: bool = field(default=False)
    max_num_batched_tokens: int = field(default=512)  # 512 is based on A100 ITL for llama model
    enable_prefix_caching: bool = field(default=False)
    vllm_command_flags: Dict[str, Optional[str]] = field(default_factory=dict)
    trust_remote_code: bool = field(default=False)
    max_model_len: Optional[int] = field(default=None)
    served_model_alias: Optional[str] = field(default=None)
    guided_decoding_backend: Optional[str] = field(default=None)

    def _to_vllm_command(self, local_model_path: Optional[str] = None) -> List[str]:
        flags = []
        skip_flags = ["--enable-chunked-prefill",
                      "--model",
                      "--max-num-batched-tokens",
                      "--enable-prefix-caching",
                      "--max-model-len",
                      "--trust-remote-code",
                      "--served-model-name",
                      "--guided-decoding-backend"]
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
                raise ValueError(f"Invalid guided decoding backend {self.guided_decoding_backend}")
            flags.append(self.guided_decoding_backend)

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

    def to_run_command(self, local_model_path: Optional[str] = None) -> List[str]:
        return self._to_vllm_command(local_model_path=local_model_path)

    def engine_pip_reqs(self, *,
                        vllm_version: str = "0.5.5",
                        httpx_version: str = "0.27.0",
                        lm_format_enforcer_version: str = "0.10.6",
                        outlines_version: str = "0.0.46") -> List[str]:
        default_installs = [f"vllm=={vllm_version}", f"httpx=={httpx_version}"]
        if self.guided_decoding_backend == "lm-format-enforcer":
            default_installs.append(f"lm-format-enforcer=={lm_format_enforcer_version}")
        if self.guided_decoding_backend == "outlines":
            default_installs.append(f"outlines=={outlines_version}")
        return default_installs


class VLLMEngineProcess(EngineProcess):

    @property
    def engine_name(self) -> str:
        return "vllm-engine"

    def health_check(self) -> bool:
        try:
            resp = self.server_http_client.get("/health")
            return resp.status_code == 200
        except Exception as e:
            debug_msg(f"Health check failed with error {e}; server may not be up yet or crashed;")
            return False
