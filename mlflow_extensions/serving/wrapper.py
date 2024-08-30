import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Type, Optional, Iterator

import mlflow
from httpx import Response
from huggingface_hub import snapshot_download

from mlflow_extensions.serving.engines.base import EngineProcess, debug_msg, EngineConfig


@dataclass
class CustomEngineServingResponse:
    status: int
    data: dict


class CustomServingEnginePyfuncWrapper(mlflow.pyfunc.PythonModel):

    # todo support lora modules
    def __init__(self,
                 *,
                 engine: Type[EngineProcess],
                 engine_config: EngineConfig,
                 model_artifact_key="model",
                 endpoint="/chat/completions"):
        self._engine_klass: Type[EngineProcess] = engine
        self._engine_config: EngineConfig = engine_config
        self._engine: Optional[EngineProcess] = None
        # for convenience
        self._model_name = self._engine_config.model
        self._model_artifact_key = model_artifact_key
        self._endpoint = endpoint
        self._artifacts = None

    @property
    def artifacts(self):
        if self._artifacts is None:
            raise ValueError("Artifacts not configured, run model.setup()")
        return self._artifacts

    @property
    def model_key(self):
        return self._model_artifact_key

    def _request_model(self, req_str):
        response = self._engine.oai_http_client.post(self._endpoint, content=req_str)
        status_code = response.status_code
        return json.dumps({"status": status_code, "data": response.text})

    @staticmethod
    def iter_mlflow_predictions(response: Response) -> Iterator[CustomEngineServingResponse]:
        mlflow_response = response.json()
        predictions = mlflow_response.get("predictions", [])
        for prediction in predictions:
            prediction = json.loads(prediction)
            data = prediction.get("data", "")
            try:
                prediction_data = json.loads(data)
            except Exception as e:
                debug_msg(f"failed to parse data; got error: {str(e)}")
                prediction_data = data
            yield CustomEngineServingResponse(
                status=prediction.get("status"),
                data=prediction_data
            )

    def load_context(self, context):
        if self._engine is None:
            self._engine = self._engine_klass(config=self._engine_config)
        model_path = context.artifacts[self._model_artifact_key]
        self._engine.start_proc(model_path)

    def predict(self, context, model_input: List[str], params=None) -> List[str]:
        return [self._request_model(req) for req in model_input]

    def _hub_download_snapshot(self, repo_name: str, local_dir: str = "/root/models"):
        local_dir = local_dir.rstrip('/')
        model_local_path = f"{local_dir}/{repo_name}"
        snapshot_download(repo_id=self._model_name,
                          local_dir=model_local_path)
        return model_local_path

    def _setup_snapshot(self, local_dir: str = "/root/models"):
        return self._hub_download_snapshot(self._model_name, local_dir)

    def _setup_artifacts(self, local_dir: str = "/root/models"):
        local_path = self._setup_snapshot(local_dir)
        self._artifacts = {self._model_artifact_key: local_path}
        return self._artifacts

    def _verify_chat_template(self,
                              artifacts: Dict[str, str],
                              tokenizer_config_file: str = "tokenizer_config.json",
                              chat_template_key: str = "chat_template",
                              ):
        model_dir_path = Path(artifacts[self._model_artifact_key])
        tokenizer_config_file = model_dir_path / tokenizer_config_file
        assert tokenizer_config_file.exists(), f"Tokenizer config file not found at {str(tokenizer_config_file)}"
        with open(str(tokenizer_config_file), "r") as f:
            tokenizer_config = json.loads(f.read())
            chat_template = tokenizer_config.get(chat_template_key)
            if chat_template is None:
                raise ValueError(f"Chat template not found in tokenizer config file {str(tokenizer_config_file)}")

    def get_pip_reqs(self, **kwargs):
        return self._engine_config.default_pip_reqs(**kwargs)

    def setup(self,
              *,
              local_dir="/root/models",
              verify_chat_template: bool = True,
              tokenizer_config_file: str = "tokenizer_config.json",
              chat_template_key: str = "chat_template"):
        self._setup_artifacts(local_dir)
        if verify_chat_template is True:
            self._verify_chat_template(self.artifacts, tokenizer_config_file, chat_template_key)
        debug_msg(f"Command to be run: {self._engine_config.to_run_command()}")
