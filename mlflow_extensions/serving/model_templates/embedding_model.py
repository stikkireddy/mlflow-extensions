from typing import List, Optional, Type

import pandas as pd
from mlflow.models import set_model
from mlflow.types.llm import ChatParams
from vllm.entrypoints.openai.protocol import EmbeddingCompletionRequest

from mlflow_extensions.serving.engines import VLLMEngineConfig, VLLMEngineProcess
from mlflow_extensions.serving.engines.base import EngineConfig, EngineProcess
from mlflow_extensions.serving.wrapper import CustomServingEnginePyfuncWrapper


class CustomServingEngineEmbeddingModel(CustomServingEnginePyfuncWrapper):

    def __init__(
        self,
        *,
        engine: Type[EngineProcess] = VLLMEngineProcess,
        engine_config: EngineConfig = {},
    ):
        self._engine_klass: Type[EngineProcess] = engine
        self._engine_config: EngineConfig = {}
        self._engine: Optional[EngineProcess] = None
        # for convenience
        self._artifacts = None

    def _request_model(self, req):

        response = self._engine.oai_http_client.post("embeddings", json=req)
        return response.json()

    def load_context(self, context):
        self._engine_config = context.model_config.get("engine_config", {})
        assert self._engine_config != {}, "Engine config must be present"
        if self._engine is None:
            self._engine = self._engine_klass(
                config=VLLMEngineConfig(**self._engine_config)
            )

        self._engine.start_proc(context)

    def predict(
        self, context, messages: List[List[str]], params: Optional[ChatParams] = None
    ):
        if not isinstance(messages, (list, dict, pd.DataFrame)):
            raise ValueError(
                f"model_input must be a list, dict but received: {type(messages)}"
            )



        if isinstance(messages, pd.DataFrame):
            assert (
                messages.shape[0] == 1
            ), "Only single multi-turn conversation should be passed"
            messages = messages.iloc[0].to_dict()

        if isinstance(messages["input"], list):
            embedding_input = messages["input"]
        else:
            embedding_input = messages["input"]

        # remove input and merge with everything else
        messages.pop("input", None)

        if params is None:
            params = {}
        params = {**params, **messages}

        request = EmbeddingCompletionRequest(
            model="default", input=embedding_input, **params
        )

        request = request.model_dump()
        if "top_logprobs" not in params:
            request.pop("logprobs", None)
            request.pop("top_logprobs", None)

        response = self._request_model(request)

        return response


model = CustomServingEngineEmbeddingModel()

set_model(model)
