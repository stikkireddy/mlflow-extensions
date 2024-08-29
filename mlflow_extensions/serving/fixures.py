import os
import signal
import subprocess
import time
from typing import Optional, List, Dict, Any

import httpx

from mlflow_extensions.serving.engines.base import debug_msg


class LocalTestServer:

    def __init__(self, *, model_uri: str,
                 registry_host: str,
                 registry_token: str,
                 test_serving_host: str = "0.0.0.0",
                 test_serving_port: int = 5000,
                 registry_is_uc: bool = False,
                 additional_serving_flags: Optional[List[str]] = None):
        self._model_uri = model_uri
        self._databricks_registry_host = registry_host
        self._databricks_registry_token = registry_token
        self._test_serving_host = test_serving_host
        self._test_serving_port = test_serving_port
        self._registry_is_uc = registry_is_uc
        self._additional_serving_flags = additional_serving_flags or []

        self._server_process = None
        self._http_client = httpx.Client(base_url=f"http://{self._test_serving_host}:{self._test_serving_port}")

    def start(self):
        try:
            subprocess.run(f"kill $(lsof -t -i:{self._test_serving_port})", shell=True)
        except Exception as e:
            debug_msg(f"Failed to kill port: {e}")
        command_args = ["mlflow", "models", "serve", "-m", self._model_uri, "-p", str(self._test_serving_port),
                        *self._additional_serving_flags]
        # spawn in new process group
        current_env = os.environ.copy()
        current_env["DATABRICKS_HOST"] = self._databricks_registry_host
        current_env["DATABRICKS_TOKEN"] = self._databricks_registry_token
        self._server_process = subprocess.Popen(command_args,
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE,
                                                preexec_fn=os.setsid,
                                                env=current_env)

    def wait_and_assert_healthy(self):
        assert self._server_process is not None, "Server process has not been started."
        while True:
            if self._server_process.returncode is not None:
                stdout, stderr = self._server_process.communicate(timeout=10)
                print('STDOUT:', stdout.decode())
                print('STDERR:', stderr.decode())
                print('Exit Code:', self._server_process.returncode)
                raise ValueError("Server process has terminated unexpectedly.")

            try:
                resp = self._http_client.get("/health")
                if resp.status_code == 200:
                    print("Success")
                    break
            except Exception as e:
                print("endpoint not yet available")
            assert self._server_process.returncode is None, "Server process has terminated unexpectedly."
            time.sleep(1)

    def query(self, *, payload: Dict[str, Any]):
        return self._http_client.post("/invocations", json=payload)

    def stop(self):
        # Send SIGTERM to the subprocess
        os.killpg(os.getpgid(self._server_process.pid), signal.SIGTERM)

        # Wait for a short while and check if the process has terminated
        time.sleep(5)
        if self._server_process.poll() is None:
            # If still running, send SIGKILL
            os.killpg(os.getpgid(self._server_process.pid), signal.SIGKILL)

        # # Optionally, wait for the process to terminate and get its exit status
        try:
            # Wait for the process to terminate, with a timeout
            stdout, stderr = self._server_process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = self._server_process.communicate()
            print("Timed out")

        print('STDOUT:', stdout.decode())
        print('STDERR:', stderr.decode())
        print('Exit Code:', self._server_process.returncode)
