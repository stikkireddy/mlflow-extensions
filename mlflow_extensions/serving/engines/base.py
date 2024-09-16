import abc
import json
import os
import random
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional, Union

import httpx
import psutil
from filelock import FileLock
from mlflow.pyfunc import PythonModelContext

from mlflow_extensions.serving.engines.gpu_utils import not_enough_shm
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
            self.active_process = subprocess.Popen(
                self.command,
                env=self.env or os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # ensure process is in another process group / session
                preexec_fn=os.setsid,
            )
        else:
            self.active_process = subprocess.Popen(
                self.command,
                env=self.env or os.environ.copy(),
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,  # This will handle the output as text rather than bytes
                bufsize=1,  # Line-buffered
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
            raise ValueError(
                "Unable to wait and log for long living process will hang the thread."
            )

        try:
            # Stream and print stdout
            for line in self.active_process.stdout:
                print(f"STDOUT: {line.strip()}")

            # Wait for the process to complete
            self.active_process.wait()

            # Check the exit code
            print("Exit Code:", self.active_process.returncode)

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
            print("STDOUT:", stdout.decode())
            print("STDERR:", stderr.decode())
        finally:
            print("Exit Code:", self.active_process.returncode)


@dataclass(frozen=True, kw_only=True)
class EngineConfig(abc.ABC):
    model: str
    host: str = field(default="0.0.0.0")
    port: int = field(default=9989)
    openai_api_path: int = field(default="v1")
    ensure_supported_models: bool = field(default=True)
    library_overrides: dict = field(default_factory=dict)

    @abc.abstractmethod
    def _to_run_command(
        self, context: PythonModelContext = None
    ) -> Union[List[str], Command]:
        pass

    def to_run_command(
        self, context: PythonModelContext = None
    ) -> Union[List[str], Command]:
        command = self._to_run_command(context)
        debug_msg(f"Command: {command}")
        if isinstance(command, list):
            # ensure all items are strings
            return [str(item) for item in command]
        return command

    def default_pip_reqs(
        self,
        *,
        filelock_version: str = "3.15.4",
        httpx_version: str = "0.27.0",
        psutil_version: str = "6.0.0",
        mlflow_extensions_version: str = None,
        **kwargs,
    ) -> List[str]:
        mlflow_extensions_version = (
            mlflow_extensions_version or get_mlflow_extensions_version()
        )
        default_installs = {
            "httpx": f"httpx=={httpx_version}",
            "psutil": f"psutil=={psutil_version}",
            "filelock": f"filelock=={filelock_version}",
            "mlflow-extensions": (
                f"mlflow-extensions=={mlflow_extensions_version}"
                if mlflow_extensions_version
                else "mlflow-extensions"
            ),
        }
        final_installs = {}
        for key, value in default_installs.items():
            final_installs[key] = value
        final_installs.update(self.engine_pip_reqs(**kwargs))
        if self.library_overrides:
            final_installs.update(self.library_overrides)
        return list(final_installs.values())

    @abc.abstractmethod
    def engine_pip_reqs(self, **kwargs) -> Dict[str, str]:
        # key should be eggname/packagename and the value should be the full
        # installation string in requirements.txt
        pass

    @abc.abstractmethod
    def setup_artifacts(self, local_dir: str = "/root/models") -> Dict[str, str]:
        pass

    @staticmethod
    @abc.abstractmethod
    def supported_model_architectures(self) -> List[str]:
        pass


class EngineHealthCheckStatusManager:

    def __init__(
        self,
        health_check_path: str = "~/.mlflow-extensions/health-check.txt",
        availability_path: str = "~/.mlflow-extensions/availability.txt",
    ):
        self._relative_path = Path(health_check_path)
        self._availability_relative_path = Path(availability_path)
        self._path = self._relative_path.expanduser().resolve()
        self._availability_path = (
            self._availability_relative_path.expanduser().resolve()
        )
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        # ensure parent directories
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._availability_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()
        if not self._availability_path.exists():
            self._availability_path.touch()

    def start_empty(self):
        with open(self._path, "w") as f:
            f.write("")

    def add_status(self, status: str):
        status = {"datetime_utc": str(datetime.now()), "status": status}
        json_status = json.dumps(status)
        with open(self._path, "a") as f:
            f.write(f"{json_status}\n")

    def set_available(self):
        with open(self._availability_path, "w") as f:
            f.write("AVAILABLE")

    def set_unavailable(self):
        with open(self._availability_path, "w") as f:
            f.write("UNAVAILABLE")

    def get_availability(self):
        with open(self._availability_path, "r") as f:
            return f.read()

    def get_last_n_status(self, n: int = 100) -> List[Dict[str, str]]:
        with open(self._path, "r") as f:
            lines = f.readlines()
        if len(lines) < n:
            return [json.loads(line) for line in lines]
        return [json.loads(line) for line in lines[-n:]]


class EngineProcess(abc.ABC):

    def __init__(self, *, config: EngineConfig):
        self._config = config
        self._lock = FileLock(f"{self.__class__.__name__}.txt.lock")
        self._server_http_client = httpx.Client(
            base_url=f"http://{self._config.host}:{self._config.port}", timeout=30
        )
        self._oai_http_client = httpx.Client(
            base_url=f"http://{self._config.host}:{self._config.port}/{self._config.openai_api_path}",
            timeout=300,
        )
        self._proc = None
        self._run_health_check = None
        self._health_check_thread = None
        self._health_check_status_file = EngineHealthCheckStatusManager()

    @property
    def server_process(self):
        return self._proc

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
            self._kill_active_proc()
            self._proc = None
            self._run_health_check = False
            self.cleanup()

    def _is_process_running(self):
        if self._proc is None:
            return False
        try:
            process = psutil.Process(self._proc.pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    def _kill_active_proc(self):
        os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        time.sleep(5)
        if self._proc.poll() is None:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)

    def cleanup(self) -> None:
        # should not throw or raise errors
        pass

    def _spawn_server_proc(self, context: PythonModelContext = None):
        proc_env = os.environ.copy()
        server_details = {"HOST": self.config.host, "PORT": str(self.config.port)}
        if not_enough_shm() is True:
            debug_msg("Not enough shared memory for NCCL. Setting NCCL_SHM_DISABLE=1")
            server_details["NCCL_SHM_DISABLE"] = "1"
        proc_env.update(server_details)
        command = self.config.to_run_command(context)
        if isinstance(command, list):
            self._proc = subprocess.Popen(
                command,
                env=proc_env,
                preexec_fn=os.setsid,
            )
        elif isinstance(command, Command):
            command.start()

        while self.is_process_healthy() is False and self._is_process_running() is True:
            debug_msg(f"Waiting for {self.engine_name} to start")
            time.sleep(1)

    def ensure_server_is_running(
        self,
        *,
        context: PythonModelContext = None,
        health_check_frequency_seconds: int = 10,
        max_respawn_attempts: int = 3,
    ):
        print(f"Ensuring {self.engine_name} is running for pid {self._proc.pid}")
        if self._proc is None:
            raise ValueError("Process not started yet. Run start_proc() first.")
        attempt_count = 0
        # check if pid is still running
        self._health_check_status_file.start_empty()
        while True:
            if attempt_count > max_respawn_attempts:
                debug_msg(
                    f"Max respawn attempts reached for {self.engine_name}. Restart serving endpoint."
                )
                self._health_check_status_file.add_status(
                    "Max respawn attempts reached, server unable to be respawned."
                )
                break
            if self._run_health_check is not True:
                break
            try:
                if self._is_process_running():
                    time.sleep(health_check_frequency_seconds)
                else:
                    self._health_check_status_file.set_unavailable()
                    self._health_check_status_file.add_status("Process is not running.")
                    process = psutil.Process(self._proc.pid)
                    if process.status() == psutil.STATUS_ZOMBIE:
                        debug_msg(
                            f"Process: {self._proc.pid} is zombie. Killing process."
                        )
                        self._health_check_status_file.add_status(
                            "Process is a zombie. Killing process."
                        )
                        self._kill_active_proc()
                    self._health_check_status_file.add_status(
                        "Process is not running. Calling cleanup."
                    )
                    self.cleanup()
                    self._health_check_status_file.add_status(
                        "Process is not running. Respawning."
                    )
                    self._spawn_server_proc(context)
                    self._health_check_status_file.add_status(
                        "Spawn Server Proc Finished."
                    )
                    if self.health_check() is False:
                        self._health_check_status_file.add_status(
                            "Health check failed after respawn. "
                            f"Attempt: {attempt_count}"
                        )
                        debug_msg(f"Health check failed after respawn.")
                        attempt_count += 1
                    else:
                        self._health_check_status_file.add_status(
                            "Health check passed after respawn."
                        )
                        self._health_check_status_file.set_available()
                        attempt_count = 0
            except psutil.NoSuchProcess:
                self._health_check_status_file.add_status(
                    "Process does not exist. Respawning."
                )
                self._spawn_server_proc(context)
                self._health_check_status_file.add_status("Spawn Server Proc Finished.")
                if self.health_check() is False:
                    self._health_check_status_file.add_status(
                        "Health check failed after respawn. "
                        f"Attempt: {attempt_count}"
                    )
                    debug_msg(f"Health check failed after respawn.")
                    attempt_count += 1
                else:
                    self._health_check_status_file.set_available()
                    self._health_check_status_file.add_status(
                        "Health check passed after respawn."
                    )
                    attempt_count = 0

    def health_check_status(self):
        return {
            "status": self._health_check_status_file.get_availability(),
            "health_check_thread_running": self._run_health_check,
            "health_check_thread_last_50_status": self._health_check_status_file.get_last_n_status(
                50
            ),
        }

    # todo add local lora paths
    def start_proc(
        self, context: PythonModelContext = None, health_check_thread: bool = True
    ):
        # kill process in port if already running
        time.sleep(random.randint(1, 5))
        debug_msg(f"Attempting to acquire Lock")
        with self._lock:
            debug_msg(f"Acquired Lock")
            if self.health_check() is False:
                self._health_check_status_file.set_unavailable()
                self._spawn_server_proc(context)
                self._health_check_status_file.set_available()
                self._run_health_check = True
                if health_check_thread is True:
                    self._health_check_thread = Thread(
                        target=self.ensure_server_is_running,
                        kwargs={
                            "context": context,
                        },
                    )
                    if self._health_check_thread is not None:
                        self._health_check_thread.start()
            else:
                debug_msg(f"{self.engine_name} already running")
