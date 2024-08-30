import abc
import os
import random
import signal
import socket
import subprocess
import time
from dataclasses import field, dataclass
from typing import List, Optional, Dict, Union

import httpx
from filelock import FileLock
from mlflow.pyfunc import PythonModelContext

from mlflow_extensions.version import get_mlflow_extensions_version


def debug_msg(msg: str):
    print(f"[DEBUG][pid:{os.getpid()}] {msg}")


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex((host, port))
        return result == 0


@dataclass(kw_only=True)
class Command:
    name: str
    command: List[str]
    active_process: Optional[subprocess.Popen] = None
    long_living: bool = True
    env: Optional[Dict[str, str]] = None

    def start(self):
        if self.long_living is True:
            self.active_process = subprocess.Popen(self.command,
                                                   env=self.env or os.environ.copy(),
                                                   stdout=subprocess.PIPE,
                                                   stderr=subprocess.PIPE,
                                                   # ensure process is in another process group / session
                                                   preexec_fn=os.setsid, )
        else:
            self.active_process = subprocess.Popen(
                self.command,
                env=self.env or os.environ.copy(),
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,  # This will handle the output as text rather than bytes
                bufsize=1  # Line-buffered
            )

    def is_running(self) -> bool:
        if self.active_process is None:
            return False
        if self.active_process.returncode is not None:
            return False
        return True

    def wait_and_log(self):
        if self.active_process is None:
            print("Process has not been started.")
            return

        if self.long_living is True:
            raise ValueError("Unable to wait and log for long living process will hang the thread.")

        try:
            # Stream and print stdout
            for line in self.active_process.stdout:
                print(f"STDOUT: {line.strip()}")

            # Wait for the process to complete
            self.active_process.wait()

            # Check the exit code
            print('Exit Code:', self.active_process.returncode)

        except Exception as e:
            print(f"Error while waiting for logs: {e}")

    def stop(self):
        if self.is_running() is False:
            return
        # Send SIGTERM to the subprocess
        os.killpg(os.getpgid(self.active_process.pid), signal.SIGTERM)

        # Wait for a short while and check if the process has terminated
        time.sleep(5)
        if self.active_process.poll() is None:
            # If still running, send SIGKILL
            os.killpg(os.getpgid(self.active_process.pid), signal.SIGKILL)

        # # Optionally, wait for the process to terminate and get its exit status
        try:
            # Wait for the process to terminate, with a timeout
            stdout, stderr = self.active_process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = self.active_process.communicate()
            print("Timed out")

        try:
            print('STDOUT:', stdout.decode())
            print('STDERR:', stderr.decode())
        finally:
            print('Exit Code:', self.active_process.returncode)


@dataclass(frozen=True, kw_only=True)
class EngineConfig(abc.ABC):
    model: str
    host: str = field(default="0.0.0.0")
    port: int = field(default=9989)
    openai_api_path: int = field(default="v1")

    @abc.abstractmethod
    def to_run_command(self, context: PythonModelContext = None) -> Union[List[str], Command]:
        pass

    def default_pip_reqs(self, *,
                         filelock_version: str = "3.15.4",
                         httpx_version: str = "0.27.0",
                         mlflow_extensions_version: str = None,
                         **kwargs) -> List[str]:

        mlflow_extensions_version = mlflow_extensions_version or get_mlflow_extensions_version()
        if mlflow_extensions_version is None:
            mlflow_extensions = "mlflow-extensions"
        else:
            mlflow_extensions = f"mlflow-extensions=={mlflow_extensions_version}"
        return [f"httpx=={httpx_version}", *self.engine_pip_reqs(**kwargs),
                f"filelock=={filelock_version}", mlflow_extensions]

    @abc.abstractmethod
    def engine_pip_reqs(self, **kwargs) -> List[str]:
        pass

    @abc.abstractmethod
    def setup_artifacts(self, local_dir: str = "/root/models") -> Dict[str, str]:
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
            timeout=300
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
    def start_proc(self, context: PythonModelContext = None):
        # kill process in port if already running
        time.sleep(random.randint(1, 5))
        debug_msg(f"Attempting to acquire Lock")
        with self._lock:
            debug_msg(f"Acquired Lock")
            if self.health_check() is False:
                proc_env = {"HOST": self.config.host, "PORT": str(self.config.host)}
                command = self.config.to_run_command(context)
                if isinstance(command, list):
                    self._proc = subprocess.Popen(self.config.to_run_command(context),
                                                  env=proc_env)
                elif isinstance(command, Command):
                    command.start()

                while self.is_process_healthy() is False:
                    debug_msg(f"Waiting for {self.engine_name} to start")
                    time.sleep(1)
            else:
                debug_msg(f"{self.engine_name} already running")
