# Databricks notebook source
# MAGIC %pip install mlflow-extensions
# MAGIC %pip install -U mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from mlflow_extensions.databricks.prebuilt import prebuilt
from mlflow_extensions.databricks.deploy.ez_deploy import EzDeploy

deployer = EzDeploy(
  config=prebuilt.vision.vllm.PHI_3_5_VISION_INSTRUCT_12K,
  registered_model_name="main.default.sri_phi_3_5_vision_instruct_12k"
)

# COMMAND ----------

deployer.download()

# COMMAND ----------

deployer.register()

# COMMAND ----------

endpoint_name = "sri_phi_3_5_vision_instruct_vllm"
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
# from pydantic import BaseModel

# class ExpectedJson(BaseModel):
#     outside: bool
#     inside: bool
#     boardwalk: bool
#     grass: bool

# while True:
response = client.chat.completions.create(
  model=my_model,
  messages=[
    {"role": "user", 
    "content": [
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
  # extra_body={
  #   "guided_choice": ["outside", "indoors"]
  # }
  # extra_body={
  #   "guided_json": ExpectedJson.schema()
  # }
)

response.choices[0].message.content.strip()

# COMMAND ----------


