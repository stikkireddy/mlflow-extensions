from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from mlflow.pyfunc import PythonModelContext

from mlflow_extensions.log import Logger, get_logger
from mlflow_extensions.serving.engines.base import Command, EngineConfig, EngineProcess
from mlflow_extensions.serving.engines.huggingface_utils import snapshot_download_local
from mlflow_extensions.testing.helper import kill_processes_containing

LOGGER: Logger = get_logger()


@dataclass(frozen=True, kw_only=True)
class InfinityEngineConfig(EngineConfig):
    batch_size: int = field(default=32)
    served_model_name: str = field(default="default")
    model_artifact_key: str = field(default="model")
    trust_remote_code: bool = field(default=False)
    model_warmup: bool = field(default=True)
    url_prefix: str = field(default="/v1")
    # compile with torch inductor
    torch_compile: bool = field(default=False)

    # custom flags
    infinity_command_flags: Dict[str, Optional[str]] = field(default_factory=dict)

    def _to_infinity_command(self, context: PythonModelContext = None) -> List[str]:
        local_model_path = None
        if context is not None:
            local_model_path = context.artifacts.get(self.model_artifact_key)

        flags = []
        skip_flags = [
            "--batch-size",
            "--served-model-name",
            "--model-id",
            "--trust-remote-code",
            "--model-warmup",
            "--url-prefix",
            "--compile",
        ]

        for k, v in self.infinity_command_flags.items():
            if k in skip_flags:
                LOGGER.info(f"Skipping flag {k} use the built in argument")
                continue
            flags.append(k)
            if v is not None:
                flags.append(v)

        if self.batch_size is not None:
            flags.append("--max-num-batched-tokens")
            flags.append(str(self.batch_size))
        if self.torch_compile is True:
            flags.append("--compile")
        if self.trust_remote_code is True:
            flags.append("--trust-remote-code")
        if self.model_warmup is True:
            flags.append("--model-warmup")
        if self.url_prefix is not None:
            flags.append("--url-prefix")
            flags.append(self.url_prefix)

        return [
            "infinity_emb",
            "v2",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--model-id",
            self.model if local_model_path is None else local_model_path,
            "--served-model-name",
            self.served_model_name,
            *flags,
        ]

    def _to_run_command(
        self, context: PythonModelContext = None
    ) -> Union[List[str], Command]:
        return self._to_infinity_command(context=context)

    def engine_pip_reqs(
        self,
        *,
        infinity_version: str = "0.0.58",
    ) -> Dict[str, str]:
        return {
            "infinity": f"infinity-emb[all]=={infinity_version}",
        }

    def _setup_snapshot(self, local_dir: str = "/local_disk0/models"):
        return snapshot_download_local(repo_id=self.model, local_dir=local_dir)

    def _setup_artifacts(self, local_dir: str = "/local_disk0/models"):
        local_path = self._setup_snapshot(local_dir)
        return {self.model_artifact_key: local_path}

    def setup_artifacts(self, local_dir: str = "/local_disk0/models"):
        artifacts = self._setup_artifacts(local_dir)
        return artifacts

    def supported_model_architectures(self) -> List[str]:
        return []


class InfinityEngineProcess(EngineProcess):

    @property
    def engine_name(self) -> str:
        return "infinity-engine"

    def health_check(self) -> bool:
        try:
            resp = self.server_http_client.get("/health")
            return resp.status_code == 200
        except Exception as e:
            LOGGER.error(
                f"Health check failed with error {e}; server may not be up yet or crashed;"
            )
            return False

    def cleanup(self) -> None:
        try:
            kill_processes_containing("infinity_emb")
        except Exception:
            pass
