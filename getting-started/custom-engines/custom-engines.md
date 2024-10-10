# Using Custom Engines

If you do not want to use any of the prebuilts this guide will help you deploy your the custom engine directly.

## Deploying Models using vLLM

vLLM is a optimized server that is optimized for running llms and multimodal lms.
It is a complex server that supports a lot of configuration/knobs to improve performance. This documentation will be
updated as we test more configurations.

### Registering a model

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

## Deploying Models using SGLang [TBD]

## Deploying Models using Ray Serving [TBD]

## Deploying Models using Ollama

Ollama is a optimized server that is optimized for running llms and multimodal lms. It supports llama.cpp as the backend
to be able to run the models using cpu and ram. This documentation will be updated as we test more configurations.

**Keep in mind databricks serving endpoints only have 4gb of memory per container.
** [Link to docs.](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-limits.html#limitations)

### Registering a model

```python

import mlflow

from mlflow_extensions.serving.engines import OllamaEngineConfig, OllamaEngineProcess
from mlflow_extensions.serving.wrapper import CustomServingEnginePyfuncWrapper

mlflow.set_registry_uri("databricks-uc")

model = CustomServingEnginePyfuncWrapper(
    engine=OllamaEngineProcess,
    engine_config=OllamaEngineConfig(
        model="gemma2:2b",
    )
)

model.setup()  # this will download ollama and the model. it may take a while so let it run.

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=model,
        artifacts=model.artifacts,
        pip_requirements=model.get_pip_reqs(),
        registered_model_name="<catalog>.<schema>.<model-name>"
    )
```