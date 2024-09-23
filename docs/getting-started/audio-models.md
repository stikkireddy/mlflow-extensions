# Audio Language Models

Audio Language models are models that are trained on audio tasks. These models can be used for various audio tasks like
summarization, transcription, etc.

You can use `EzDeploy` or `EzDeployLite` to deploy these models to model serving. Read the previous guides.
For the scope of this we will be using `EzDeployLite` to deploy a audio model.

## EzDeployLite

```python
%pip install mlflow-extensions==0.14.0
dbutils.library.restartPython()

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployLite
from mlflow_extensions.databricks.prebuilt import prebuilt

deployer = EzDeployLite(
    ez_deploy_config=prebuilt.audio.vllm.FIXIE_ULTRA_VOX_0_4_64K_CONFIG
)

deployment_name = "my_ultravox_model"
deployer.deploy(deployment_name)
```

The code will deploy a audio model to Databricks jobs and expose the model as via a proxy.
This is meant for dev and testing use cases.

## Querying using OpenAI SDK for Audio Models

### Using Audio File URLs

You can query the model using image URLs. The model will return the desired output based on the image provided.

Installation:

```python
%pip install mlflow-extensions==0.14.0
%pip install -U openai
dbutils.library.restartPython()
```

Query Model:

```python
from mlflow_extensions.serving.compat import get_ezdeploy_lite_openai_url

deployment_name = "my_ultravox_model"
base_url = get_ezdeploy_lite_openai_url(deployment_name)

from openai import OpenAI
from mlflow.utils.databricks_utils import get_databricks_host_creds

client = OpenAI(base_url=base_url, api_key=get_databricks_host_creds().token)
for i in client.models.list():
    model = i.id

print(model)

response = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Is the image indoors or outdoors?"},
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg.wav"
                    },
                },
            ],
        }
    ],
)
print(response)
```

### Using Audio clips from base64 encoded strings

You can also query the model using images from the file system. The model will return the desired output based on the
image provided.

Installation:

```python
%pip install mlflow-extensions==0.14.0
%pip install -U openai
dbutils.library.restartPython()
```

Download Image into Base64; you can modify this to fetch from file system but the logic would be the same.

Query Model:

```python
import base64
import requests
from mlflow_extensions.serving.compat import get_ezdeploy_lite_openai_url

deployment_name = "my_qwen_model"
base_url = get_ezdeploy_lite_openai_url(deployment_name)

from openai import OpenAI
from mlflow.utils.databricks_utils import get_databricks_host_creds

client = OpenAI(base_url=base_url, api_key=get_databricks_host_creds().token)
for i in client.models.list():
    model = i.id

print(model)


## Use base64 encoded image in the payload
def encode_audio_base64_from_url(audio_url: str) -> str:
    """Encode an image retrieved from a remote url to base64 format."""

    # you can modify this with reading an image from a file
    with requests.get(audio_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')

    return result


# gettysburg.wav is a 17 second audio file
audio_data = encode_audio_base64_from_url("https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg.wav")
chat_completion_from_base64 = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What’s in this image?"
            },
            {
                "type": "audio_url",
                "audio_url": {
                    # Any format supported by librosa is supported
                    "url": f"data:audio/ogg;base64,{audio_data}"
                },
            },
        ],
    }],
    max_tokens=256,
)
print(chat_completion_from_base64)
```