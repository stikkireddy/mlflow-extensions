# Databricks notebook source
# MAGIC %pip install mlflow-extensions
# MAGIC %pip install mlflow -U
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeploy
from mlflow_extensions.databricks.prebuilt import prebuilt

deployer = EzDeploy(
    config=prebuilt.audio.vllm.FIXIE_ULTRA_VOX_0_4_64K_CONFIG,
    registered_model_name="main.default.sri_ultravox_audio_text_to_text_model"
)

deployer.download()

deployer.register()

endpoint_name = "sri_ultravox_audio_text_to_text_model"

deployer.deploy(endpoint_name)

# COMMAND ----------

import requests
import base64


def encode_audio_base64_from_url(audio_url: str) -> str:
    """Encode an audio retrieved from a remote url to base64 format."""

    with requests.get(audio_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')

    return result


# gettysburg.wav is a 17 second audio file
audio_data = encode_audio_base64_from_url("https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg.wav")

# COMMAND ----------

from mlflow_extensions.serving.compat.openai import OpenAI
from mlflow_extensions.databricks.prebuilt import prebuilt
from mlflow.utils.databricks_utils import get_databricks_host_creds

workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
endpoint_name = "sri_ultravox_audio_text_to_text_model"
endpoint_url = f"https://{workspace_host}/serving-endpoints/{endpoint_name}/invocations"

token = get_databricks_host_creds().token

client = OpenAI(
    base_url=endpoint_url,
    api_key=token
)

model_name = prebuilt.audio.vllm.FIXIE_ULTRA_VOX_0_4_64K_CONFIG.engine_config.model

chat_completion_from_base64 = client.chat.completions.create(
    messages=[{
        "role":
            "user",
        "content": [
            {
                "type": "text",
                "text": "Breakdown the content of the audio?"
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
    model=model_name,
    max_tokens=512,
)

result = chat_completion_from_base64.choices[0].message.content
print(result)

# COMMAND ----------

from mlflow_extensions.serving.compat.openai import OpenAI
from mlflow_extensions.databricks.prebuilt import prebuilt
from mlflow.utils.databricks_utils import get_databricks_host_creds

workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
endpoint_name = "sri_ultravox_audio_text_to_text_model"
endpoint_url = f"https://{workspace_host}/serving-endpoints/{endpoint_name}/invocations"

token = get_databricks_host_creds().token

client = OpenAI(
    base_url=endpoint_url,
    api_key=token
)

model_name = prebuilt.audio.vllm.FIXIE_ULTRA_VOX_0_4_64K_CONFIG.engine_config.model

from pydantic import BaseModel
from typing import Literal, List


class AudioExtraction(BaseModel):
    year: str
    speaker: str
    location: str
    sentiment: Literal["positive", "negative", "neutral"]
    tone: Literal["somber", "upbeat", "pessemistic"]
    summary: str


chat_completion_from_base64 = client.chat.completions.create(
    messages=[{
        "role":
            "user",
        "content": [
            {
                "type": "text",
                "text": f"Extract the following content from the audio: {str(AudioExtraction.schema())}?"
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
    model=model_name,
    max_tokens=512,
    extra_body={
        "guided_json": AudioExtraction.schema()
    }
)

result = chat_completion_from_base64.choices[0].message.content
print(result)

# COMMAND ----------


