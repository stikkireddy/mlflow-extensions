# mlflow extensions

## Overview


The goal of this project is to make deploying any large language model, or multi modal large language models 
a simple three-step process.

1. Download the model from hf or any other source.
2. Register the model with mlflow.
3. Deploy the model using the mlflow serving infrastructure. (e.g. Databricks)


## Features

1. Testing pyfunc models using `mlflow_extensions.serving.fixtures.LocalTestServer` in Databricks notebooks.
2. Deploying vision models, etc using `mlflow_extensions.serving.engines.vllm_engine` in Databricks model serving.
3. Deploy models using cpu via ollama engine.

## Installation

```bash
pip install mlflow-extensions
```

## Supported Server Frameworks

1. vLLM
2. Ollama 
3. SGlang

## EZ Deploy

To make your deployments easier into a three step process we have created a simplified interface that lets you 
download the model and then register in UC and deploy it in Databricks with the appropriate gpu hardware.

<span style="color: red; font-weight: bold;">
[AS OF SEPT 4, 2024] IF YOU ARE DEPLOYING MODELS INTO HARDWARE WITH MULTIPLE GPUS AT THE MOMENT SHM (SHARED ACCESS MEMORY) IS LIMITED IN GPU CONTAINERS TO 64MB DEFAULT. PLEASE REACH OUT TO YOUR DATABRICKS ACCOUNT TEAM IF PERFORMANCE IS IMPACTING YOU TO HAVE THIS INCREASED. THIS IS A KNOWN LIMIT OF THE CONTAINERS AND THIS FRAMEWORK DISABLES NCCL USAGE OF SHM.
</span>

Out of the box Ez Deploy Models: 

**Note this framework supports much larger set of models these are the ones that have been curated and validated.**

| model_type   | cfg_path                                            | huggingface_link                                          | context_length   | min_azure_ep_type_gpu          | min_aws_ep_type_gpu           |
|:-------------|:----------------------------------------------------|:----------------------------------------------------------|:-----------------|:-------------------------------|:------------------------------|
| text         | prebuilt.text.sglang.GEMMA_2_9B_IT                  | https://huggingface.co/google/gemma-2-9b-it               | Default          | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB] |
| text         | prebuilt.text.vllm.NUEXTRACT                        | https://huggingface.co/numind/NuExtract                   | Default          | GPU_LARGE [A100_80Gx1 80GB]    | GPU_MEDIUM [A10Gx1 24GB]      |
| text         | prebuilt.text.vllm.NUEXTRACT_TINY                   | https://huggingface.co/numind/NuExtract-tiny              | Default          | GPU_SMALL [T4x1 16GB]          | GPU_SMALL [T4x1 16GB]         |
| text         | prebuilt.text.vllm.NOUS_HERMES_3_LLAMA_3_1_8B_64K   | https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B | 64000            | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB] |
| text         | prebuilt.text.vllm.NOUS_HERMES_3_LLAMA_3_1_8B_128K  | https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B | Default          | GPU_LARGE_2 [A100_80Gx2 160GB] | GPU_MEDIUM_8 [A10Gx8 192GB]   |
| vision       | prebuilt.vision.sglang.LLAVA_1_5_LLAMA              | https://huggingface.co/liuhaotian/llava-v1.5-7b           | Default          | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB] |
| vision       | prebuilt.vision.sglang.LLAVA_1_6_VICUNA             | https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b    | Default          | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB] |
| vision       | prebuilt.vision.sglang.LLAVA_1_6_YI_34B             | https://huggingface.co/liuhaotian/llava-v1.6-34b          | Default          | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB] |
| vision       | prebuilt.vision.sglang.LLAVA_NEXT_LLAMA3_8B         | https://huggingface.co/lmms-lab/llama3-llava-next-8b      | Default          | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB] |
| vision       | prebuilt.vision.vllm.PHI_3_5_VISION_INSTRUCT_4K     | https://huggingface.co/microsoft/Phi-3.5-vision-instruct  | 4096             | GPU_SMALL [T4x1 16GB]          | GPU_SMALL [T4x1 16GB]         |
| vision       | prebuilt.vision.vllm.PHI_3_5_VISION_INSTRUCT_8K     | https://huggingface.co/microsoft/Phi-3.5-vision-instruct  | 8192             | GPU_SMALL [T4x1 16GB]          | GPU_SMALL [T4x1 16GB]         |
| vision       | prebuilt.vision.vllm.PHI_3_5_VISION_INSTRUCT_12K    | https://huggingface.co/microsoft/Phi-3.5-vision-instruct  | 12000            | GPU_LARGE [A100_80Gx1 80GB]    | GPU_MEDIUM [A10Gx1 24GB]      |
| vision       | prebuilt.vision.vllm.PHI_3_5_VISION_INSTRUCT_32K    | https://huggingface.co/microsoft/Phi-3.5-vision-instruct  | 32000            | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB] |
| vision       | prebuilt.vision.vllm.PHI_3_5_VISION_INSTRUCT_64K    | https://huggingface.co/microsoft/Phi-3.5-vision-instruct  | 64000            | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB] |
| vision       | prebuilt.vision.vllm.PHI_3_5_VISION_INSTRUCT_128K   | https://huggingface.co/microsoft/Phi-3.5-vision-instruct  | Default          | GPU_LARGE_2 [A100_80Gx2 160GB] | GPU_MEDIUM_8 [A10Gx8 192GB]   |
| audio        | prebuilt.audio.vllm.FIXIE_ULTRA_VOX_0_4_64K_CONFIG  | https://huggingface.co/fixie-ai/ultravox-v0_4             | 64000            | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB] |
| audio        | prebuilt.audio.vllm.FIXIE_ULTRA_VOX_0_4_128K_CONFIG | https://huggingface.co/fixie-ai/ultravox-v0_4             | Default          | GPU_LARGE_2 [A100_80Gx2 160GB] | GPU_MEDIUM_8 [A10Gx8 192GB]   |

### Deploying a model using EZ Deploy

Look at [01-getting-started-phi-3.5-vision-instruct.py](examples%2Fnotebooks%2F01-getting-started-phi-3.5-vision-instruct.py) 
for a complete example using phi 3.5 vision limited to a 12k context window running properly on a model serving endpoint

```python

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeploy
from mlflow_extensions.databricks.prebuilt import prebuilt

deployer = EzDeploy(
  config=prebuilt.vision.vllm.PHI_3_5_VISION_INSTRUCT_12K,
  registered_model_name="main.default.phi_3_5_vision_instruct_12k"
)

deployer.download()

deployer.register()

endpoint_name = "my-endpoint-name"

deployer.deploy("my-endpoint-name")


model_name = prebuilt.vision.vllm.PHI_3_5_VISION_INSTRUCT_12K.engine_config.model

from mlflow_extensions.serving.compat.openai import OpenAI
from mlflow.utils.databricks_utils import get_databricks_host_creds

workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
endpoint_name = f"https://{workspace_host}/serving-endpoints/{endpoint_name}/invocations"
token = get_databricks_host_creds().token

client = OpenAI(
  base_url=endpoint_name,
  api_key=token
)

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": "Hi how are you?"
        }
    ],
)
```


## Custom Engine Usage

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

# assert local_server.query(payload={"inputs": [....]}) == ...

local_server.stop()
```

### Deploying Models using Ollama 

Ollama is a optimized server that is optimized for running llms and multimodal lms. It supports llama.cpp as the backend
to be able to run the models using cpu and ram. This documentation will be updated as we test more configurations.

**Keep in mind databricks serving endpoints only have 4gb of memory per container.** [Link to docs.](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-limits.html#limitations)

#### Registering a model

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

model.setup() # this will download ollama and the model. it may take a while so let it run.

with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            "model",
            python_model=model,
            artifacts=model.artifacts,
            pip_requirements=model.get_pip_reqs(),
            registered_model_name="<catalog>.<schema>.<model-name>"
        )
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

### Calling models using openai compatible clients

#### Calling a model using openai sdk with basic completion

Mlflow extensions offers a wrapper on top of openai sdk to intercept requests and conform them to model serving infra.

Supported engines:
- [x] vLLM
- [x] Ollama

```python
from mlflow_extensions.serving.compat.openai import OpenAI
# if you need async client
# from mlflow_extensions.serving.compat.openai import AsyncOpenAI

client = OpenAI(base_url="https://<>.com/serving-endpoints/<model-name>", api_key="<dapi...>")
response = client.chat.completions.create(
  model="gemma2:2b",
  messages=[
    {"role": "user", "content": "Hi how are you?"}
  ],
)
```

#### Calling a model using openai sdk that supports multi modal inputs (vision)

Supported engines:
- [x] vLLM
- [x] Ollama

Mlflow extensions offers a wrapper on top of openai sdk to intercept requests and conform them to model serving infra.

```python
from mlflow_extensions.serving.compat.openai import OpenAI

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
)
```

#### Guided decoding into json

Make sure you deploy a model with guided_decoding_backend configured. 
The proper values are either outlines or lm-format-enforcer. **Currently only supported by VLLMEngine.**

```python
from mlflow_extensions.serving.compat.openai import OpenAI
from pydantic import BaseModel

class Data(BaseModel):
  outside: bool
  inside: bool

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
  extra_body={
    "guided_json": Data.schema()
  }
  #   if you want to use guided choice to select one of the choices
  # extra_body={
  #   "guided_choice": ["outside", "indoors"]
  # }
)
```

#### Calling a model using langchain ChatOpenAI sdk

```python
from mlflow_extensions.serving.compat.langchain import ChatOpenAI
# if you want to use completions
# from mlflow_extensions.serving.compat.langchain import OpenAI

model = ChatOpenAI(
    model="gemma2:2b",
    base_url="https://<>.com/serving-endpoints/<model-name>", 
    api_key="<dapi...>"
)
model.invoke("hello world")
```

#### Calling a model using sglang sdk using the openai backend

```python
from sglang import function, system, user, assistant, gen, set_default_backend
from mlflow_extensions.serving.compat.sglang import OpenAI


@function
def multi_turn_question(s, question_1, question_2):
    s += system("You are a helpful assistant.")
    s += user(question_1)
    s += assistant(gen("answer_1", max_tokens=256))
    s += user(question_2)
    s += assistant(gen("answer_2", max_tokens=256))

set_default_backend(
    OpenAI(
        model="gemma2:2b",
        base_url="https://<>.com/serving-endpoints/<model-name>",
        api_key="<dapi..."
    )
)
    
state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions there.",
    )

for m in state.messages():
    print(m["role"], ":", m["content"])

print("answer 1", state["answer_1"])
print("answer 2", state["answer_2"])
```


#### Calling a model using sglang sdk using the sglang built-in backend

```python
from sglang import function, system, user, assistant, gen, set_default_backend
from mlflow_extensions.serving.compat.sglang import RuntimeEndpoint


@function
def multi_turn_question(s, question_1, question_2):
    s += system("You are a helpful assistant.")
    s += user(question_1)
    s += assistant(gen("answer_1", max_tokens=256))
    s += user(question_2)
    s += assistant(gen("answer_2", max_tokens=256))

set_default_backend(
    RuntimeEndpoint(
        "https://<>.com/serving-endpoints/<model-name>",
        "<dapi..."
    )
)
    
state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions there.",
    )

for m in state.messages():
    print(m["role"], ":", m["content"])

print("answer 1", state["answer_1"])
print("answer 2", state["answer_2"])
```

#### Supported engines

##### vLLM engine

Here are the list of supported models for vllm engine: https://docs.vllm.ai/en/latest/models/supported_models.html

We have not tested all of them please raise a issue if there is one that does not work. 
We will work on documenting models and configs. Please document the model, size, and config you used to deploy 
where you ran into issues.

##### Ollama engine

Here are the list of supported models for ollama. [Link to model list.](https://ollama.com/library)

**Keep in mind databricks serving endpoints only have 4gb of memory per container.** [Link to docs.](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-limits.html#limitations)

## Hardware Diagnostics

### GPU Diagnostics

TBD

## Optimizations Roadmap

1. Prefix Caching Enablement for ez deploy based on task type (some flag like repeated long prompt)
2. Speculative Decoding Enablement [ngram based] for ez deploy based on task type (some flag like data extraction)
3. Quantized Models curated from huggingface
4. Quantized KV Cache support

## Disclaimer
mlflow-extensions is not developed, endorsed not supported by Databricks. It is provided as-is; no warranty is derived from using this package. 
For more details, please refer to the license.
