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
# assert ez_deploy_config, "ez_deploy_config is required"
# assert pip_reqs, "pip_reqs is required"

# COMMAND ----------

import re

pattern = r"mlflow-extensions==[^\s]+"
temp_fix_url = "git+https://github.com/changshilim-db/mlflow-extensions.git@fix/gcp_singapore"

# Remove 'mlflow-extensions==<version>' from the string
pip_reqs = re.sub(pattern, '', pip_reqs).strip()
pip_reqs += " git+https://github.com/changshilim-db/mlflow-extensions.git@fix/gcp_singapore"
print(pip_reqs)

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

# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
if hf_secret_scope and hf_secret_key:
    os.environ["HF_TOKEN"] = dbutils.secrets.get(
        scope=hf_secret_scope, key=hf_secret_key
    )

# COMMAND ----------

import socket
if os.getenv("HOST_IP", ""):
    print(f'HOST_IP is assigned: {os.getenv("HOST_IP")}')
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))  
    vllm_ip = s.getsockname()[0]
    os.environ['VLLM_HOST_IP'] = vllm_ip

from vllm.utils import get_ip
print(f'vLLM IP: {get_ip()}')

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

from dbruntime.databricks_repl_context import get_context
ctx = get_context()

url = f'https://{ctx.browserHostName}'
url = url.rstrip("/")
cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
print(
    "Base URL for this endpoint:", f"{url}/driver-proxy-api/o/0/{cluster_id}/9989/v1/"
)

# COMMAND ----------

import time

while True:
    time.sleep(1)
