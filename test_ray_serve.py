# Databricks notebook source
# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployLite
from mlflow_extensions.databricks.prebuilt import prebuilt

# COMMAND ----------

deployer = EzDeployLite(
  ez_deploy_config=prebuilt.vision.vllm.QWEN2_VL_7B_INSTRUCT
)

# COMMAND ----------

deployment_name = "my_qwen_model"
deployer.deploy(deployment_name, specific_git_ref="branch/main")

# COMMAND ----------


