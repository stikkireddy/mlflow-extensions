# Databricks notebook source
dbutils.widgets.text("ez_deploy_config", "")
dbutils.widgets.text("hf_secret_scope", "")
dbutils.widgets.text("hf_secret_key", "")
dbutils.widgets.text("pip_reqs", "")

# COMMAND ----------

ez_deploy_config = dbutils.widgets.get("ez_deploy_config")
hf_secret_scope = dbutils.widgets.get("hf_secret_scope")
hf_secret_key = dbutils.widgets.get("hf_secret_key")
pip_reqs = dbutils.widgets.get("pip_reqs")
assert ez_deploy_config, "ez_deploy_config is required"
assert pip_reqs, "pip_reqs is required"

# COMMAND ----------

from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic("pip", f"install {pip_reqs} hf_transfer")
ipython.run_line_magic("pip", f"install -U openai")
dbutils.library.restartPython()

# COMMAND ----------

ez_deploy_config = dbutils.widgets.get("ez_deploy_config")
hf_secret_scope = dbutils.widgets.get("hf_secret_scope")
hf_secret_key = dbutils.widgets.get("hf_secret_key")
pip_reqs = dbutils.widgets.get("pip_reqs")
assert ez_deploy_config, "ez_deploy_config is required"
assert pip_reqs, "pip_reqs is required"

# COMMAND ----------

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
if hf_secret_scope and hf_secret_key:
    os.environ["HF_TOKEN"] = dbutils.secrets.get(
        scope=hf_secret_scope, key=hf_secret_key
    )

# COMMAND ----------

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployConfig

config = EzDeployConfig.from_json(ez_deploy_config)

# COMMAND ----------

engine_process = config.to_proc()

# COMMAND ----------

artifacts = config.download_artifacts()

# COMMAND ----------

from mlflow.pyfunc import PythonModelContext

ctx = PythonModelContext(artifacts=artifacts, model_config={})

# COMMAND ----------

import ray

from mlflow_extensions.testing.helper import kill_processes_containing

ray.shutdown()
kill_processes_containing("vllm")
kill_processes_containing("ray")
kill_processes_containing("sglang")
kill_processes_containing("from multiprocessing")
engine_process.start_proc(ctx, health_check_thread=False)

# COMMAND ----------

import time

while True:
    time.sleep(1)
