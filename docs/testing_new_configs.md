# New Config Testing or Server Testing

This document is a guide to testing new configurations or servers. It is important to test new configurations or servers 
before deploying them to production. This document will guide you through the process of testing new configurations or 
servers.

## Integration Testing New Configurations

To test new configurations, follow these steps:

1. Make sure you are using 15.4 LTS ML with an A100 GPU or 4xA10 GPUs for any model < 70b fp16 parameters. Use 2xA100 for models > 70b fp16 parameters.
2. Construct the config and engines
3. Test running the process and make sure you dont have health thread running
4. Test the process with a simple model
5. Stop the process or you will have to kill all processes running on the port

Engine configurations can be found in:
1. vllm: `mlflow_extensions/serving/engines/vllm_engine.py` 
2. sglang: `mlflow_extensions/serving/engines/sglang_engine.py`

If you need flags from the server itself you can find them here:
1. vllm: https://docs.vllm.ai/en/latest/models/engine_args.html

You can use these additional args that are not built in to the `VLLMEngineConfig` model like so:

```python
from mlflow_extensions.serving.engines import VLLMEngineConfig
VLLMEngineConfig(
    model="...",
    vllm_command_flags={
        # args with actual values
        "--arg": "value",
        # flag that are truthy
        "--flag": None
    }
)
```

### Example with vllm testing

Here is an example with vLLM but Sglang can be tested in a similar way.

```python
from mlflow.pyfunc import PythonModelContext
from mlflow_extensions.serving.engines import VLLMEngineProcess, VLLMEngineConfig
from mlflow_extensions.testing.helper import kill_processes_containing
from openai import OpenAI

# kill any existing vllm processes
kill_processes_containing("vllm.entrypoints.openai.api_server")
config = VLLMEngineConfig(
        model="NousResearch/Hermes-3-Llama-3.1-8B",
        trust_remote_code=True,
        guided_decoding_backend="outlines",
        max_model_len=64000,
    )

artifacts = config.setup_artifacts()

mlflow_ctx = PythonModelContext(
  artifacts=artifacts,
  model_config={}
)

nuextract_engine = VLLMEngineProcess(
    config=config
)

nuextract_engine.start_proc(
    context=mlflow_ctx,
    health_check_thread=False # make sure this is false it will keep spawning server if it shuts down
)

client = OpenAI(
  base_url=f"http://{config.host}:{config.port}/v1", 
  api_key="foo")

response = client.chat.completions.create(
  model=config.model,
  messages=[
    {"role": "system","content": "You are a helpful assistant."},
    {"role": "user","content": "Why is the sky blue?"}
  ],
  max_tokens=512
)
print(response.choices[0].message.content)

# shut down model using
nuextract_engine.stop_proc()
```


## Building an EzDeploy Config

To build a EzDeploy Config you need to go to the following folders for the appropriate modality:
1. audio: `mlflow_extensions/databricks/prebuilt/audio`
2. text: `mlflow_extensions/databricks/prebuilt/text`
3. vision: `mlflow_extensions/databricks/prebuilt/vision`

Please use the existing EzDeploy configs for reference and look at where the fields are being used.

An EzDeploy Config needs to look like the following:

```python
from mlflow_extensions.databricks.deploy.ez_deploy import (
    EzDeployConfig,
    ServingConfig,
)
from mlflow_extensions.serving.engines import VLLMEngineProcess, VLLMEngineConfig

_ENGINE = VLLMEngineProcess
_ENGINE_CONFIG = VLLMEngineConfig

NEW_NOUS_CONFIG = EzDeployConfig(
    # needs a name
    name="hermes_3_llama_3_1_8b_64k",
    engine_proc=_ENGINE,
    # the appropriate configs
    engine_config=_ENGINE_CONFIG(
        model="NousResearch/Hermes-3-Llama-3.1-8B",
        trust_remote_code=True,
        guided_decoding_backend="outlines",
        max_model_len=64000,
    ),
    # the serving config, either estimated memory or specific gpus
    serving_config=ServingConfig(
        # rough estimate for the engines this includes model weights + kv cache + overhead + intermediate states
        minimum_memory_in_gb=60,
    ),
)
```

In the previous code example we made: `NEW_NOUS_CONFIG` which is a `EzDeployConfig` object. 
We can add that to the registry by going to the bottom and looking for:

```python
from dataclasses import dataclass
@dataclass(frozen=True)
class VllmText:
    ...
```

Then register the new model:

```python
from dataclasses import dataclass, field
@dataclass(frozen=True)
class VllmText:
     NEW_NOUS_CONFIG = field(default_factory=lambda: NEW_NOUS_CONFIG)
```

Then go run the tests in the `mlflow_extensions/tests/integration/vllm` or `mlflow_extensions/tests/integration/sglang` 
folder to make sure the config is correct.