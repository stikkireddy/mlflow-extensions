# mlflow extensions

## Overview

The goal of this project is to extend the capabilities of MLflow to support additional features such as 
testing pyfunc in databricks notebooks, and deploy complex llm server infrastructure such as vllm, sglang, ollama, etc.

## Features

1. Testing pyfunc models using `mlflow_extensions.serving.fixtures.LocalTestServer` in Databricks notebooks.
2. Deploying vision models, etc using `mlflow_extensions.serving.engines.vllm_engine` in Databricks model serving.

## Installation

```bash
pip install mlflow-extensions
```

## Supported Server Frameworks

1. vLLM
2. SGlang (tbd)
3. Ollama (tbd)

## Usage

### Testing Pyfunc Models

The local test server will spawn a local server that will serve the model and can be queried using the `query` method.
It will spawn the server in its own process group id and if you need to control the port, test_serving_port can be passed.

```python
from mlflow_extensions.serving.fixures import LocalTestServer
from mlflow.utils.databricks_utils import get_databricks_host_creds


local_server = LocalTestServer(
  model_uri="<uri to the model or run>",
  registry_host=get_databricks_host_creds().host,
  registry_token=get_databricks_host_creds().token
)

local_server.start()

local_server.wait_and_assert_healthy()

# assert fixture.query(payload={"inputs": [....]}) == ...

local_server.stop()
```

### Deploying Models using vLLM 

vLLM is a optimized server that is optimized for running llms and multimodal lms. 
It is a complex server that supports a lot of configuration/knobs to improve performance. This documentation will be
updated as we test more configurations.


#### Registering a model

```python
import mlflow

from mlflow_extensions.serving.engines import VLLMEngineProcess, VLLMEngineConfig
from mlflow_extensions.serving.wrapper import CustomServingEnginePyfuncWrapper

mlflow.set_registry_uri("databricks-uc")

# optionally if you need to download model from hf which is not public facing
# os.environ["HF_TOKEN"] = ...

model = CustomServingEnginePyfuncWrapper(
    engine=VLLMEngineProcess,
    engine_config=VLLMEngineConfig(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,
        max_model_len=64000,  # max token length for context
        guided_decoding_backend="outlines"
    )
)

model.setup()  # download artifacts from huggingface

with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            "model",
            python_model=model,
            artifacts=model.artifacts,
            pip_requirements=model.get_pip_reqs(),
            registered_model_name="<catalog>.<schema>.<model-name>"
        )
```

#### Calling a model using openai sdk

Mlflow extensions offers a wrapper on top of openai sdk to intercept requests and conform them to model serving infra.

```python
from mlflow_extensions.serving.adapters import OpenAIWrapper as OpenAI

client = OpenAI(base_url="https://<>.com/serving-endpoints/<model-name>", api_key="<dapi...>")
response = client.chat.completions.create(
  model="microsoft/Phi-3.5-vision-instruct",
  messages=[
    {"role": "user", "content": [
                {"type": "text", "text": "Is the image indoors or outdoors?"},
                {
                    "type": "image_url",
                    "image_url": {
                      "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    },
                },
            ],
     }
  ],
  #   if you want to use guided decoding to improve performance and control output
  # extra_body={
  #   "guided_choice": ["outside", "indoors"]
  # }
)
```

#### Calling a model using langchain ChatOpenAI sdk

```
from mlflow_extensions.serving.adapters import ChatOpenAIWrapper as ChatOpenAI

model = ChatOpenAI(base_url="https://<>.com/serving-endpoints/<model-name>", api_key="<dapi...>")
model.invoke("hello world")
```

#### Supported models

##### vLLM engine

Here are the list of supported models for vllm engine: https://docs.vllm.ai/en/latest/models/supported_models.html

We have not tested all of them please raise a issue if there is one that does not work. 
We will work on documenting models and configs. Please document the model, size, and config you used to deploy 
where you ran into issues.

## Disclaimer
mlflow-extensions is not developed, endorsed not supported by Databricks. It is provided as-is; no warranty is derived from using this package. 
For more details, please refer to the license.
