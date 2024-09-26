# Databricks notebook source
# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

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
deployer.deploy(deployment_name, specific_git_ref="branch/main")

# COMMAND ----------

deployer._elhm

# COMMAND ----------


