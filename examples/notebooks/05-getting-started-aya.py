# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC # KEEP IN MIND AYA does not have a apache 2.0 or MIT license it is cc-by-nc-4.0 
# MAGIC
# MAGIC # YOU MUST adhere to C4AI's Acceptable Use Policy
# MAGIC
# MAGIC Read more here: https://huggingface.co/CohereForAI/aya-23-35B

# COMMAND ----------

# MAGIC %pip install mlflow-extensions
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
  config=prebuilt.text.vllm.COHERE_FOR_AYA_23_35B,
  registered_model_name="main.default.cohere_aya_35b"
)

deployer.download()

deployer.register()

endpoint_name = "sri_cohere_aya"

deployer.deploy(endpoint_name)

# COMMAND ----------

endpoint_name = "sri_cohere_aya"

from mlflow_extensions.serving.compat.openai import OpenAI
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow_extensions.databricks.prebuilt import prebuilt

workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
endpoint_name = f"https://{workspace_host}/serving-endpoints/{endpoint_name}/invocations"
token = get_databricks_host_creds().token

client = OpenAI(
  base_url=endpoint_name,
  api_key=token
)

response = client.chat.completions.create(
    model=prebuilt.text.vllm.COHERE_FOR_AYA_23_35B.engine_config.model,
    messages=[
        {
            "role": "user",
            "content": "Hi what model are you, who trained you?"
        }
    ],
)
response

# COMMAND ----------


