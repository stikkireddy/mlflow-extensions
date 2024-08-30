import abc
import os
import random
import socket
import subprocess
import time
from dataclasses import field, dataclass
from typing import List, Optional

import httpx
from filelock import FileLock

from mlflow_extensions.version import get_mlflow_extensions_version


def debug_msg(msg: str):
    print(f"[DEBUG][pid:{os.getpid()}] {msg}")


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex((host, port))
        return result == 0


@dataclass(frozen=True, kw_only=True)
class EngineConfig(abc.ABC):
    model: str
    host: str = field(default="0.0.0.0")
    port: int = field(default=9989)
    openai_api_path: int = field(default="v1")

    @abc.abstractmethod
    def to_run_command(self, local_model_path: Optional[str] = None) -> List[str]:
        pass

    def default_pip_reqs(self, *,
                         filelock_version: str = "3.15.4",
                         mlflow_extensions_version: str = None,
                         **kwargs) -> List[str]:

        mlflow_extensions_version = mlflow_extensions_version or get_mlflow_extensions_version()
        if mlflow_extensions_version is None:
            mlflow_extensions = "mlflow-extensions"
        else:
            mlflow_extensions = f"mlflow-extensions=={mlflow_extensions_version}"
        return [*self.engine_pip_reqs(**kwargs), f"filelock=={filelock_version}", mlflow_extensions]

    @abc.abstractmethod
    def engine_pip_reqs(self, **kwargs) -> List[str]:
        pass


class EngineProcess(abc.ABC):

    def __init__(self, *, config: EngineConfig):
        self._config = config
        self._lock = FileLock(f"{self.__class__.__name__}.txt.lock")
        self._server_http_client = httpx.Client(
            base_url=f"http://{self._config.host}:{self._config.port}",
            timeout=30
        )
        self._oai_http_client = httpx.Client(
            base_url=f"http://{self._config.host}:{self._config.port}/{self._config.openai_api_path}",
            timeout=30
        )
        self._proc = None

    @property
    def oai_http_client(self) -> httpx.Client:
        return self._oai_http_client

    @property
    def server_http_client(self) -> httpx.Client:
        return self._server_http_client

    @property
    @abc.abstractmethod
    def engine_name(self) -> str:
        pass

    @property
    def config(self) -> EngineConfig:
        return self._config

    @abc.abstractmethod
    def health_check(self) -> bool:
        pass

    def is_process_healthy(self) -> bool:
        if not is_port_open(self.config.host, self.config.port):
            return False

        return self.health_check()

    def stop_proc(self):
        with self._lock:
            if self.is_process_healthy() is True:
                subprocess.run(f"kill $(lsof -t -i:{self.config.port})", shell=True)

    # todo add local lora paths
    def start_proc(self, local_model_path: Optional[str] = None):
        # kill process in port if already running
        time.sleep(random.randint(1, 5))
        debug_msg(f"Attempting to acquire Lock")
        with self._lock:
            debug_msg(f"Acquired Lock")
            if self.health_check() is False:
                proc_env = {"HOST": self.config.host, "PORT": str(self.config.host)}
                self._proc = subprocess.Popen(self.config.to_run_command(local_model_path),
                                              env=proc_env)
                while self.is_process_healthy() is False:
                    debug_msg(f"Waiting for {self.engine_name} to start")
                    time.sleep(1)
            else:
                debug_msg(f"{self.engine_name} already running")
