# Databricks notebook source
# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# %load_ext autoreload
# %autoreload 2

# COMMAND ----------

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployLite,EzDeployRayServe
from mlflow_extensions.databricks.prebuilt import prebuilt

# COMMAND ----------

deployer = EzDeployRayServe(
  ez_deploy_config=prebuilt.text.vllm.QWEN2_5_14B_INSTRUCT)

# COMMAND ----------

from mlflow_extensions.databricks.deploy.ez_deploy_lite import JobsConfig
prebuilt.text.vllm.QWEN2_5_14B_INSTRUCT.engine_config.

# COMMAND ----------

deployment_name = "QWEN2_5_14B"
deployer.deploy(deployment_name, specific_git_ref="branch/ray_serve_check", min_replica = 2,max_replica =2)

# COMMAND ----------

from mlflow_extensions.databricks.deploy.ez_deploy_ray_serve import make_create_json

create_json = make_create_json(
    job_name="job_name",
    minimum_memory_in_gb=deployer._config.serving_config.minimum_memory_in_gb,
    Replica = 1,
    cloud_provider=deployer._cloud,
    ez_deploy_config=deployer._config,
    huggingface_secret_scope=None,
    huggingface_secret_key=None,
    specific_git_ref="branch/main",
)

# COMMAND ----------

deployer._edlm.client.jobs.create(**create_json)

# COMMAND ----------

print(create_json['tasks')

# COMMAND ----------


