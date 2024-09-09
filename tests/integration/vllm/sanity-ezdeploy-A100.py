# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # These are sanity tests for ezdeploy for vLLM on A100

# COMMAND ----------

# MAGIC %pip install vllm==0.6.0 filelock httpx hf_transfer
# MAGIC %pip install -U openai
# MAGIC %pip install -U mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip list | grep vllm

# COMMAND ----------

from mlflow_extensions.databricks.deploy.gpu_configs import AzureServingGPUConfig
from mlflow_extensions.testing.runner import (
    RequestResult,
    ServerFramework,
    run_all_tests,
)

THIS_GPU = AzureServingGPUConfig.GPU_LARGE.value
THIS_FRAMEWORK = ServerFramework.VLLM

# COMMAND ----------

import os

os.environ["HF_TOKEN"] = dbutils.secrets.get(
    scope="sri-mlflow-extensions", key="hf-token"
)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# COMMAND ----------

results = run_all_tests(gpu_config=THIS_GPU, server_framework=THIS_FRAMEWORK)

# COMMAND ----------

import pandas as pd

display(pd.DataFrame(RequestResult.make_df_friendly(results)))

# COMMAND ----------

# MAGIC %md
# MAGIC # FAILING TESTS

# COMMAND ----------

import pandas as pd

# errored records
errored_results = [result for result in results if result.is_error is True]
if len(errored_results) > 0:
    display(pd.DataFrame(RequestResult.make_df_friendly(errored_results)))
else:
    print("No tests failed...")

# COMMAND ----------

# MAGIC %md
# MAGIC # Success Records

# COMMAND ----------

import pandas as pd

# errored records
success_results = [result for result in results if result.is_error is False]
if len(success_results) > 0:
    display(pd.DataFrame(RequestResult.make_df_friendly(success_results)))
else:
    print("No tests passed...")

# COMMAND ----------

assert len(errored_results) == 0, "Tests failed"

# COMMAND ----------
