import json
import os
import signal
import subprocess
import time
import typing
from threading import Thread
from typing import Optional, List, Dict, Any

import httpx
import queue

from mlflow_extensions.serving.engines.base import debug_msg
from mlflow_extensions.serving.serde_v2 import MlflowPyfuncHttpxSerializer

if typing.TYPE_CHECKING:
    from openai import OpenAI


class FixedSizeLogQueue(queue.Queue):
    def __init__(self, max_size: int = 10000):
        # support at most 10000 log-lines
        super().__init__(maxsize=max_size)

    def put(self, item, block=True, timeout=None):
        if self.full():
            self.get_nowait()
        super().put(item, block, timeout)


class LocalTestServer:

    def __init__(
        self,
        *,
        model_uri: str,
        registry_host: str,
        registry_token: str,
        test_serving_host: str = "0.0.0.0",
        test_serving_port: int = 5000,
        registry_is_uc: bool = False,
        additional_serving_flags: Optional[List[str]] = None,
    ):
        self._model_uri = model_uri
        self._databricks_registry_host = registry_host
        self._databricks_registry_token = registry_token
        self._test_serving_host = test_serving_host
        self._test_serving_port = test_serving_port
        self._registry_is_uc = registry_is_uc
        self._additional_serving_flags = additional_serving_flags or []

        self._server_process = None
        self._http_client = httpx.Client(
            base_url=f"http://{self._test_serving_host}:{self._test_serving_port}"
        )
        self._log_queue = FixedSizeLogQueue(max_size=10000)

    def start(self):
        try:
            subprocess.run(f"kill $(lsof -t -i:{self._test_serving_port})", shell=True)
        except Exception as e:
            debug_msg(f"Failed to kill port: {e}")
        command_args = [
            "mlflow",
            "models",
            "serve",
            "-m",
            self._model_uri,
            "-p",
            str(self._test_serving_port),
            *self._additional_serving_flags,
        ]
        # spawn in new process group
        current_env = os.environ.copy()
        current_env["DATABRICKS_HOST"] = self._databricks_registry_host
        current_env["DATABRICKS_TOKEN"] = self._databricks_registry_token
        self._server_process = subprocess.Popen(
            command_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
            env=current_env,
        )
        Thread(
            target=self._enqueue_output,
            args=(self._server_process.stdout, self._log_queue),
        ).start()
        Thread(
            target=self._enqueue_output,
            args=(self._server_process.stderr, self._log_queue),
        ).start()

    def _enqueue_output(self, pipe, q: queue.Queue):
        for line in iter(pipe.readline, b""):
            q.put(line)
        pipe.close()

    def _flush_current_logs(self, max_wait_time_seconds: int = 10):
        time_start = time.time()

        while True:
            # ensure that we get to other parts of the code
            if time.time() - time_start > max_wait_time_seconds:
                break
            try:
                line = self._log_queue.get_nowait()
                data = line.decode().strip()
                if len(data) > 0:
                    print(data)
            except queue.Empty:
                break

    def wait_and_assert_healthy(self):
        assert self._server_process is not None, "Server process has not been started."
        while True:
            try:
                # try to update the returncode incase server crashes
                self._server_process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass

            self._flush_current_logs()

            if self._server_process.returncode is not None:
                stdout, stderr = self._server_process.communicate(timeout=10)
                print("STDOUT:", stdout.decode())
                print("STDERR:", stderr.decode())
                print("Exit Code:", self._server_process.returncode)
                raise ValueError("Server process has terminated unexpectedly.")

            try:
                resp = self._http_client.get("/health")
                if resp.status_code == 200:
                    print("Success")
                    break
            except Exception as e:
                print(
                    f"[HEALTH_CHECK] endpoint not yet available; health check error {str(e)}"
                )
            assert (
                self._server_process.returncode is None
            ), "Server process has terminated unexpectedly."

    def query(self, *, payload: Dict[str, Any], timeout: int = 30):
        return self._http_client.post("/invocations", json=payload, timeout=timeout)

    def query_custom_server(
        self,
        *,
        method: str,
        http_path: str,
        headers: dict = None,
        api_payload: Dict[str, Any] = None,
        timeout: int = 30,
        is_openai_compatible: bool = False,
    ):
        self._flush_current_logs()
        orig_request = httpx.Request(
            method=method,
            url=f"http://{self._test_serving_host}:{self._test_serving_port}{http_path}",
            headers=headers or {},
            content=json.dumps(api_payload) if api_payload else None,
        )
        response = self._http_client.post(
            "/invocations",
            json={
                "inputs": [
                    MlflowPyfuncHttpxSerializer.serialize_request(
                        request=orig_request,
                        url_path_to_request=http_path,
                        requires_openai_compat=is_openai_compatible,
                    )
                ]
            },
            timeout=timeout,
        )
        if response.status_code != 200:
            print(response.json())
        predictions = response.json()["predictions"]
        self._flush_current_logs()
        return MlflowPyfuncHttpxSerializer.deserialize_response(
            predictions[0], orig_request
        )

    @property
    def openai_client(self) -> "OpenAI":
        from mlflow_extensions.serving.compat.openai import OpenAI

        return OpenAI(
            base_url=f"http://{self._test_serving_host}:{self._test_serving_port}/invocations",
            api_key="foobar",
        )

    def stop(self):
        self._flush_current_logs()
        os.killpg(os.getpgid(self._server_process.pid), signal.SIGTERM)
        time.sleep(5)
        if self._server_process.poll() is None:
            os.killpg(os.getpgid(self._server_process.pid), signal.SIGKILL)
        try:
            stdout, stderr = self._server_process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = self._server_process.communicate()
            print("Timed out")

        print("STDOUT:", stdout.decode())
        print("STDERR:", stderr.decode())
        print("Exit Code:", self._server_process.returncode)
