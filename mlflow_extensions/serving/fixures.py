import json
import os
import signal
import subprocess
import time
import typing
from typing import Optional, List, Dict, Any

import httpx

from mlflow_extensions.serving.engines.base import debug_msg
from mlflow_extensions.serving.serde_v2 import MlflowPyfuncHttpxSerializer

if typing.TYPE_CHECKING:
    from openai import OpenAI


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
            try:
                # try to update the returncode incase server crashes
                self._server_process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass

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
                print(f"endpoint not yet available; health check error {str(e)}")
            assert self._server_process.returncode is None, "Server process has terminated unexpectedly."

    def query(self, *, payload: Dict[str, Any], timeout: int = 30):
        return self._http_client.post("/invocations", json=payload, timeout=timeout)

    def query_custom_server(self, *,
                            method: str,
                            http_path: str,
                            headers: dict = None,
                            api_payload: Dict[str, Any] = None,
                            timeout: int = 30,
                            is_openai_compatible: bool = False):
        orig_request = httpx.Request(
            method=method,
            url=f"http://{self._test_serving_host}:{self._test_serving_port}{http_path}",
            headers=headers or {},
            content=json.dumps(api_payload) if api_payload else None
        )
        response = self._http_client.post("/invocations", json={
            "inputs": [MlflowPyfuncHttpxSerializer.serialize_request(
                request=orig_request,
                url_path_to_request=http_path,
                requires_openai_compat=is_openai_compatible
            )]
        }, timeout=timeout)
        if response.status_code != 200:
            print(response.json())
        predictions = response.json()["predictions"]
        return MlflowPyfuncHttpxSerializer.deserialize_response(predictions[0], orig_request)

    @property
    def OpenAI(self) -> "OpenAI":
        from mlflow_extensions.serving.compat.openai import OpenAI
        return OpenAI(base_url=f"http://{self._test_serving_host}:{self._test_serving_port}/invocations",
                      api_key="foobar")

    def stop(self):
        os.killpg(os.getpgid(self._server_process.pid), signal.SIGTERM)
        time.sleep(5)
        if self._server_process.poll() is None:
            os.killpg(os.getpgid(self._server_process.pid), signal.SIGKILL)
        try:
            stdout, stderr = self._server_process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = self._server_process.communicate()
            print("Timed out")

        print('STDOUT:', stdout.decode())
        print('STDERR:', stderr.decode())
        print('Exit Code:', self._server_process.returncode)
