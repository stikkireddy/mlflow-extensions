# Databricks notebook source
# MAGIC %pip install mlflow-extensions==0.12.0
# MAGIC %pip install -U mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeploy
from mlflow_extensions.databricks.prebuilt import prebuilt
import os

os.environ["HF_TOKEN"] = dbutils.secrets.get(
    scope="sri-mlflow-extensions", key="hf-token"
)

deployer = EzDeploy(
  config=prebuilt.vision.vllm.QWEN2_VL_7B_INSTRUCT,
  registered_model_name="main.default.qwen2_vl_7b_instruct"
)

deployer.download()

deployer.register()

endpoint_name = "sri_qwen2_vl_7b_instruct"

deployer.deploy(endpoint_name, scale_to_zero=False)

# COMMAND ----------

endpoint_name = "sri_qwen2_vl_7b_instruct"

from mlflow_extensions.serving.compat.openai import OpenAI
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow_extensions.databricks.prebuilt import prebuilt
from pydantic import BaseModel
import typing as t

workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
endpoint_name = f"https://{workspace_host}/serving-endpoints/{endpoint_name}/invocations"
token = get_databricks_host_creds().token

client = OpenAI(
  base_url=endpoint_name,
  api_key=token
)


class Comparison(BaseModel):
  image_1_details: str
  image_2_details: str
  image_1_colors: t.List[str]
  image_2_colors: t.List[str]
  image_1_has_human: bool
  image_2_has_human: bool
  image_1_human_gender: t.Literal["male", "female", "no human"]
  image_2_human_gender: t.Literal["male", "female", "no human"]

prompt = f"Compare the two images and use this schema for reference and no markdown just valid json: {Comparison.schema()}"



url_1 = "https://richmedia.ca-richimage.com/ImageDelivery/imageService?profileId=12026540&id=1859027&recipeId=728"
url_2 = "https://m.media-amazon.com/images/I/81W3YQdu-tL._AC_SY550_.jpg"

response = client.chat.completions.create(
    model=prebuilt.vision.vllm.QWEN2_VL_7B_INSTRUCT.engine_config.model,
    messages=[
      {
        "role": "user",
        "content": [
            {"type" : "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": url_1}},
            {"type" : "text", "text": "to this image. Answer in english."},
            {"type": "image_url", "image_url": {"url": url_2}}
        ]
      }
    ],
    max_tokens=8192,
)
print(response.choices[0].message.content)

# COMMAND ----------


