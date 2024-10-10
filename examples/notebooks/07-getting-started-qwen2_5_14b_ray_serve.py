# Databricks notebook source
# MAGIC %pip install mlflow-extensions
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployRayServe
from mlflow_extensions.databricks.prebuilt import prebuilt

# COMMAND ----------

deployer = EzDeployRayServe(
    ez_deploy_config=prebuilt.text.vllm.QWEN2_5_14B_INSTRUCT)

# COMMAND ----------

deployment_name = "QWEN2_5_14B"
deployer.deploy(deployment_name,
                min_replica=1,
                max_replica=1)
