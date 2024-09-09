# Databricks notebook source
# MAGIC %pip install mlflow-extensions
# MAGIC %pip install -U mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from mlflow_extensions.databricks.deploy.ez_deploy import (
    EzDeployConfig,
    ServingConfig,
)
from mlflow_extensions.serving.engines import VLLMEngineProcess, VLLMEngineConfig

_ENGINE = VLLMEngineProcess
_ENGINE_CONFIG = VLLMEngineConfig


MISTRAL_LOCAL_CONFIG = EzDeployConfig(
    name="mistral_7B_instruct",
    engine_proc=_ENGINE,
    engine_config=_ENGINE_CONFIG(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        cache_dir="/Volumes/forrest_murray/models/mistral/", # can also be dbfs location
        trust_remote_code=False,
        guided_decoding_backend="outlines",
    ),
    serving_config=ServingConfig(
        # rough estimate for the engines this includes model weights + kv cache + overhead + intermediate states
        minimum_memory_in_gb=110,
    ),
)

# COMMAND ----------

from mlflow_extensions.databricks.prebuilt import prebuilt
from mlflow_extensions.databricks.deploy.ez_deploy import EzDeploy

deployer = EzDeploy(
  config=MISTRAL_LOCAL_CONFIG,
  registered_model_name="forrest_murray.models.mistral_7b_instruct_v02"
)

# COMMAND ----------

deployer.download()

# COMMAND ----------

deployer.register()

# COMMAND ----------

endpoint_name = "forrest_mistral_7b_instruct_v02"
deployer.deploy(endpoint_name)

# COMMAND ----------

dbutils.notebook.exit()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Wait for model to deploy
# MAGIC

# COMMAND ----------

from mlflow_extensions.serving.compat.openai import OpenAI
from mlflow.utils.databricks_utils import get_databricks_host_creds

workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
endpoint_name = f"https://{workspace_host}/serving-endpoints/{endpoint_name}/invocations"
token = get_databricks_host_creds().token

# COMMAND ----------

client = OpenAI(
  base_url=endpoint_name,
  api_key=token
)

my_model = None
for model in client.models.list():
  print(model.id)
  my_model = model.id

# COMMAND ----------

# Try this if you want to do guided decoding
from pydantic import BaseModel

class ExpectedJson(BaseModel):
    about_rat: bool

# while True:
response = client.chat.completions.create(
  model=my_model,
  messages=[
    {"role": "user", 
    "content": [
                {"type": "text", "text": "Remy, a young rat with heightened senses of taste and smell, dreams of becoming a chef like his human idol, the late Auguste Gusteau, but the rest of his colony, including his older brother Émile and his father, the clan leader Django, only eat for sustenance and are wary of humans."},
            ],
    }
  ],
  # extra_body={
  #   "guided_choice": ["outside", "indoors"]
  # }
  extra_body={
    "guided_json": ExpectedJson.schema()
  }
)

response.choices[0].message.content.strip()

# COMMAND ----------


