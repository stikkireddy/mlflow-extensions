from dataclasses import dataclass
from pathlib import Path
from typing import List, Type, Optional, Iterator

import mlflow
import pandas as pd
from httpx import Response, Request
from mlflow.pyfunc import PythonModelContext

from mlflow_extensions.serving.engines.base import (
    EngineProcess,
    debug_msg,
    EngineConfig,
)
from mlflow_extensions.serving.serde import ResponseMessageV1
from mlflow_extensions.serving.serde_v2 import MlflowPyfuncHttpxSerializer


@dataclass
class CustomEngineServingResponse:
    status: int
    data: dict


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
        response = self._engine.oai_http_client.send(req)
        return MlflowPyfuncHttpxSerializer.serialize_response(response)

    @staticmethod
    def iter_mlflow_predictions(response: Response) -> Iterator[ResponseMessageV1]:
        mlflow_response = response.json()
        predictions = mlflow_response.get("predictions", [])
        for prediction in predictions:
            yield ResponseMessageV1.deserialize(prediction)

    def load_context(self, context: PythonModelContext):
        if self._engine is None:
            self._engine = self._engine_klass(config=self._engine_config)
        self._engine.start_proc(context)

    def predict(
        self, context, model_input: List[List[str]], params=None
    ) -> List[List[str]]:
        import numpy as np

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
        return [
            self._request_model(
                MlflowPyfuncHttpxSerializer.deserialize_request(
                    req,
                    openai_base_url=self._engine.oai_http_client.base_url,
                    server_base_url=self._engine.server_http_client.base_url,
                )
            )
            for req in model_input
        ]

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
        debug_msg(f"Setting up artifacts in {home_directory}")
        self._setup_artifacts(str(home_directory))
        debug_msg(f"Command to be run: {self._engine_config.to_run_command()}")
