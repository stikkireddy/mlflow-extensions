import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Type

import mlflow
import numpy as np
import pandas as pd
from httpx import Request, Response
from mlflow.pyfunc import PythonModelContext

from mlflow_extensions.log import LogConfig, Logger, get_logger, initialize_logging
from mlflow_extensions.serving.compute_details import get_compute_details
from mlflow_extensions.serving.engines.base import EngineConfig, EngineProcess
from mlflow_extensions.serving.serde import ResponseMessageV1
from mlflow_extensions.serving.serde_v2 import (
    MlflowPyfuncHttpxSerializer,
    make_error_response,
)

LOGGER: Logger = get_logger()


@dataclass
class CustomEngineServingResponse:
    status: int
    data: dict


DIAGNOSTICS_REQUEST_KEY = "COMPUTE_DIAGNOSTICS"
ENABLE_DIAGNOSTICS_FLAG = "ENABLE_DIAGNOSTICS"
HEALTH_CHECK_KEY = "HEALTH_CHECK"

LOG_LEVEL: str = "LOG_LEVEL"
LOG_FILE_KEY: str = "LOG_FILE"
ARCHIVE_LOG_PATH_KEY: str = "ARCHIVE_LOG_PATH"


class CustomServingEnginePyfuncWrapper(mlflow.pyfunc.PythonModel):

    # todo support lora modules
    def __init__(self, *, engine: Type[EngineProcess], engine_config: EngineConfig):
        self._engine_klass: Type[EngineProcess] = engine
        self._engine_config: EngineConfig = engine_config
        self._engine: Optional[EngineProcess] = None
        # for convenience
        self._model_name = self._engine_config.model
        self._artifacts = None

    @property
    def artifacts(self):
        if self._artifacts is None:
            raise ValueError("Artifacts not configured, run model.setup()")
        return self._artifacts

    def _request_model(self, req: Request):
        # deserializer takes care of identifying whether oai client is needed or standard server client
        # so directly send to the root of the server
        try:
            response = self._engine.server_http_client.send(req)
        except Exception as e:
            LOGGER.error(
                f"Error in request: {req}, server_status: {self._engine.health_check_status()},"
                f" error: {str(e)}"
            )
            return MlflowPyfuncHttpxSerializer.serialize_response(
                make_error_response(
                    original_request=req,
                    error_message=str(e),
                    error_type=str(type(e)),
                    error_details={
                        "server_status": self._engine.health_check_status(),
                        "nvidia-smi": get_compute_details(cmd_key="nvidia-smi"),
                        "active-procs": get_compute_details(cmd_key="active-procs"),
                        "command": self._engine_config.to_run_command(),
                    },
                )
            )
        return MlflowPyfuncHttpxSerializer.serialize_response(response)

    @staticmethod
    def iter_mlflow_predictions(response: Response) -> Iterator[ResponseMessageV1]:
        mlflow_response = response.json()
        predictions = mlflow_response.get("predictions", [])
        for prediction in predictions:
            yield ResponseMessageV1.deserialize(prediction)

    def load_context(self, context: PythonModelContext):
        default_home_path = "~/.mlflow-extensions/logs/serving.log"
        resolved_path = os.path.expanduser(default_home_path)
        log_config: LogConfig = LogConfig(
            filename=os.environ.get(LOG_FILE_KEY, str(resolved_path)),
            archive_path=os.environ.get(ARCHIVE_LOG_PATH_KEY),
        )
        initialize_logging(log_config)

        if self._engine is None:
            self._engine = self._engine_klass(config=self._engine_config)

        self._engine.start_proc(context)

    def predict(
        self, context, model_input: List[List[str]], params=None
    ) -> List[List[str]]:
        if not isinstance(model_input, (list, dict, np.ndarray, pd.DataFrame)):
            raise ValueError(
                f"model_input must be a list, dict, numpy array, or dataframe but received: {type(model_input)}"
            )
        if isinstance(model_input, dict):
            model_input = model_input.values()
        if isinstance(model_input, pd.DataFrame):
            # check if its one column
            if len(model_input.columns) == 1:
                model_input = model_input[model_input.columns[0]].values
            else:
                raise ValueError(
                    f"Dataframe must have only one column, but received {len(model_input.columns)} columns"
                )

        responses = []
        for req in model_input:
            req: str
            if (
                req.startswith(DIAGNOSTICS_REQUEST_KEY)
                and os.environ.get(ENABLE_DIAGNOSTICS_FLAG, "false") == "true"
            ):
                parts = req.split(":")
                if len(parts) > 1:
                    compute_details = get_compute_details(parts[1])
                else:
                    compute_details = get_compute_details(cmd_key="all")
                compute_details.update(
                    {
                        "command": self._engine_config.to_run_command(context),
                    }
                )
                responses.append(json.dumps(compute_details))
            elif (
                req.startswith(DIAGNOSTICS_REQUEST_KEY)
                and os.environ.get(ENABLE_DIAGNOSTICS_FLAG, "false") == "false"
            ):
                responses.append(
                    f"Diagnostics are disabled please set environment variable on your deployment "
                    f"ENABLE_DIAGNOSTICS=true to enable them."
                )
            elif req.startswith(HEALTH_CHECK_KEY):
                responses.append(json.dumps(self._engine.health_check_status()))
            else:
                responses.append(
                    self._request_model(
                        MlflowPyfuncHttpxSerializer.deserialize_request(
                            req,
                            openai_base_url=self._engine.oai_http_client.base_url,
                            server_base_url=self._engine.server_http_client.base_url,
                        )
                    )
                )
        return responses

    def _setup_artifacts(self, local_dir: str = "/root/models"):
        self._artifacts = self._engine_config.setup_artifacts(local_dir)
        return self._artifacts

    def get_pip_reqs(self, **kwargs):
        return self._engine_config.default_pip_reqs(**kwargs)

    def setup(self, *, local_dir=None):
        if local_dir is None:
            home_directory = Path.home() / "models"
            home_directory.mkdir(parents=True, exist_ok=True)
        else:
            home_directory = local_dir
        LOGGER.info(f"Setting up artifacts in {home_directory}")
        self._setup_artifacts(str(home_directory))
        LOGGER.info(f"Command to be run: {self._engine_config.to_run_command()}")
