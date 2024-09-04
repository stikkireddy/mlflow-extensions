# Databricks notebook source
# MAGIC %pip install mlflow-extensions
# MAGIC %pip install mlflow -U
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeploy
from mlflow_extensions.databricks.prebuilt import prebuilt

deployer = EzDeploy(
  config=prebuilt.text.vllm.NUEXTRACT,
  registered_model_name="main.default.nuextract_vllm"
)

deployer.download()

deployer.register()

endpoint_name = "nuextract_vllm"

deployer.deploy(endpoint_name)


# COMMAND ----------

from mlflow_extensions.serving.compat.openai import OpenAI
from mlflow_extensions.databricks.prebuilt import prebuilt
from mlflow.utils.databricks_utils import get_databricks_host_creds

workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
endpoint_name = "nuextract_vllm"
endpoint_url = f"https://{workspace_host}/serving-endpoints/{endpoint_name}/invocations"

token = get_databricks_host_creds().token

client = OpenAI(
  base_url=endpoint_url,
  api_key=token
)

model_name = prebuilt.text.vllm.NUEXTRACT.engine_config.model

from pydantic import BaseModel
from typing import Literal, List

class ExtractedBody(BaseModel):
    product: str
    languages: Literal["python", "sql", "scala"]
    keywords: List[str]
    strategies: List[str]


response = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": """The Databricks Lakehouse Platform for Dummies is your guide to simplifying 
your data storage. The lakehouse platform has SQL and performance 
capabilities - indexing, caching and MPP processing - to make 
BI work rapidly on data lakes. It also provides direct file access 
and direct native support for Python, data science and 
AI frameworks without the need to force data through an 
SQL-based data warehouse. Find out how the lakehouse platform 
creates an opportunity for you to accelerate your data strategy."""
        }
    ],
    extra_body={
        "guided_json": ExtractedBody.schema()
    }
)
response.choices[0].message.content

# COMMAND ----------


