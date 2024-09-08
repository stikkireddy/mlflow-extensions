# Databricks notebook source
# MAGIC %pip install -U "mlflow-extensions>=0.9.0,<1.0.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow.pyfunc

class CustomModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        # raise ValueError("some bug getting artifacts or importing stuff")
        return

    def predict(self, context, model_input, params=None):
        return model_input*2

# COMMAND ----------

with mlflow.start_run() as run:
    model = CustomModel()
    mlflow.pyfunc.log_model(
        "model",
        python_model=model,
    )

# COMMAND ----------

run_uri = f"runs:/{run.info.run_id}/model"
run_uri

# COMMAND ----------

import numpy as np
loaded_model = mlflow.pyfunc.load_model(run_uri)
# context, model_input, params (default None)
first_attempt = loaded_model.predict([[1, 2, 3]])
second_attempt = loaded_model.predict(np.array([1, 2, 3]))
first_attempt, second_attempt

# COMMAND ----------

from mlflow_extensions.testing.fixures import LocalTestServer
from mlflow.utils.databricks_utils import get_databricks_host_creds

server_configs = {
  "model_uri": run_uri,
  "registry_host": get_databricks_host_creds().host,
  "registry_token": get_databricks_host_creds().token,
  "use_local_env": True
}

with LocalTestServer(**server_configs) as server:
    resp = server.query(payload={
      "inputs": [1, 2, 3]
    }).json()
    print(resp)
    assert resp == {'predictions': [2, 4, 6]}, "Predictions should be double of the input"

# COMMAND ----------

from mlflow_extensions.testing.fixures import LocalTestServer
from mlflow.utils.databricks_utils import get_databricks_host_creds

server_configs = {
  "model_uri": run_uri,
  "registry_host": get_databricks_host_creds().host,
  "registry_token": get_databricks_host_creds().token,
  "use_local_env": True
}

server = LocalTestServer(**server_configs)

server.start()

server.wait_and_assert_healthy()

resp = server.query(payload={
  "inputs": [1, 2, 3]
}).json()

print(resp)

assert resp == {'predictions': [2, 4, 6]}, "Predictions should be double of the input"

server.stop()
