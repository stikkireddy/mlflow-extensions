from dataclasses import dataclass
from typing import Dict, List, Union

from mlflow.pyfunc import PythonModelContext

from mlflow_extensions.log import Logger, get_logger
from mlflow_extensions.serving.engines.base import Command, EngineConfig, EngineProcess

LOGGER: Logger = get_logger()


@dataclass(frozen=True, kw_only=True)
class MockEngineConfig(EngineConfig):
    def _to_run_command(
        self, context: PythonModelContext = None
    ) -> Union[List[str], Command]:
        return [
            "uvicorn",
            "mlflow_extensions.serving.mock.server:app",
            "--host",
            self.host,
            "--port",
            self.port,
        ]

    def engine_pip_reqs(self, **kwargs) -> Dict[str, str]:
        return {"uvicorn": "uvicorn", "fastapi": "fastapi"}

    def setup_artifacts(self, local_dir: str = "/root/models") -> Dict[str, str]:
        return {}

    def supported_model_architectures(self) -> List[str]:
        return []


class MockEngineProcess(EngineProcess):
    @property
    def engine_name(self) -> str:
        return "mock-engine"

    def health_check(self) -> bool:
        try:
            resp = self.server_http_client.get("/health")
            return resp.status_code == 200
        except Exception as e:
            LOGGER.info(
                f"Health check failed with error {e}; server may not be up yet or crashed;"
            )
            return False
