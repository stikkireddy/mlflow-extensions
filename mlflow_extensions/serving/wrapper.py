import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Type, Optional, Iterator

import mlflow
from httpx import Response
from mlflow.pyfunc import PythonModelContext

from mlflow_extensions.serving.engines.base import EngineProcess, debug_msg, EngineConfig
from mlflow_extensions.serving.serde import RequestMessageV1, ResponseMessageV1


@dataclass
class CustomEngineServingResponse:
    status: int
    data: dict


class CustomServingEnginePyfuncWrapper(mlflow.pyfunc.PythonModel):

    # todo support lora modules
    def __init__(self,
                 *,
                 engine: Type[EngineProcess],
                 engine_config: EngineConfig):
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

    def _request_model(self, req: RequestMessageV1):
        response = self._engine.oai_http_client.request(
            method=req.method,
            url=req.request_path,
            timeout=req.timeout,
            content=req.payload
        )
        status_code = response.status_code
        return ResponseMessageV1(
            request_method=req.method,
            request_timeout=req.timeout,
            response_data=response.text,
            response_status_code=status_code,
            response_content_type=response.headers.get("Content-Type", "")
        ).serialize()

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

    def predict(self, context, model_input: List[List[str]], params=None) -> List[List[str]]:
        import numpy as np
        if not isinstance(model_input, (list, dict, np.ndarray)):
            raise ValueError(f"model_input must be a list or dict but received: {type(model_input)}")
        if isinstance(model_input, dict):
            model_input = model_input.values()
        return [self._request_model(RequestMessageV1.deserialize(req)) for req in model_input]

    def _setup_artifacts(self, local_dir: str = "/root/models"):
        self._artifacts = self._engine_config.setup_artifacts(local_dir)
        return self._artifacts

    def get_pip_reqs(self, **kwargs):
        return self._engine_config.default_pip_reqs(**kwargs)

    @staticmethod
    def get_signature():
        req = RequestMessageV1(
            timeout=1,
            payload="{}",
            method="GET",
            request_path="/chat/completions"
        ).serialize()
        resp = ResponseMessageV1(
            request_method="GET",
            request_timeout=1,
            response_data="{}",
            response_status_code=200,
            response_content_type="application/json"
        ).serialize()
        import numpy as np
        return mlflow.models.infer_signature(
            model_input=np.array([req]),
            model_output=np.array([resp])
        )

    def setup(self, *, local_dir=None):
        if local_dir is None:
            home_directory = Path.home() / "models"
            home_directory.mkdir(parents=True, exist_ok=True)
        else:
            home_directory = local_dir
        debug_msg(f"Setting up artifacts in {home_directory}")
        self._setup_artifacts(str(home_directory))
        debug_msg(f"Command to be run: {self._engine_config.to_run_command()}")
