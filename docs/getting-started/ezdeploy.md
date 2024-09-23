# Deploying a model with EzDeploy

## Requirements

1. You need access to model serving in your region
2. Your region needs to support gpus (T4, A10, A100, or H100)
3. You need to have access to any compute to run the script from a notebook. (Serverless or interactive)
4. Access to unity catalog schema to register the model.

## What is EzDeploy?

EzDeploy will take a prebuilt configuration and deploy it to databricks model serving.
This is meant for production use cases. It will support either vLLM or SGLang as engines.

## Deployment Steps

### 1. Install the library

```python
%pip install mlflow-extensions
dbutils.library.restartPython()
```

### 2. Identify the model to deploy

In this scenario we will deploy a Nous Hermes model to model serving.

```python
from mlflow_extensions.databricks.deploy.ez_deploy import EzDeploy
from mlflow_extensions.databricks.prebuilt import prebuilt

deployer = EzDeploy(
    # The model config to deploy
    config=prebuilt.text.vllm.NOUS_HERMES_3_LLAMA_3_1_8B_64K,
    # The model to register in unity catalog
    registered_model_name="main.default.nous_research_hermes_3_1"
)

deployer.download()

deployer.register()

# Deploy the model to model serving using the following endpoint name
endpoint_name = "my-endpoint-name"

deployer.deploy(endpoint_name)
```

### 3. Monitor the deployment

You will receive an url for the model serving endpoint. Monitor that url to see the status of the deployment.

## Querying using OpenAI SDK

The models are deployed as a pyfunc and they do not support natural json and need to fit the pyfunc spec. To allow you
to
use OpenAI, langchain, etc. we offer a compatability interface for those clients.

```python
from mlflow_extensions.serving.compat.openai import OpenAI
from mlflow.utils.databricks_utils import get_databricks_host_creds

workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
endpoint_name = f"https://{workspace_host}/serving-endpoints/{endpoint_name}/invocations"
token = get_databricks_host_creds().token

# this is imported from mlflow_extensions.serving.compat.openai
client = OpenAI(
    base_url=endpoint_name,
    api_key=token
)

response = client.chat.completions.create(
    # models will have their own name and will also have an alias called "default"
    model="default",
    messages=[
        {
            "role": "user",
            "content": "Hi how are you?"
        }
    ],
)
```

## Querying using Langchain SDK

You can also use query the data using ChatOpenAI using langchain sdk.

```python
from mlflow_extensions.serving.compat.langchain import ChatOpenAI

# if you want to use completions
# from mlflow_extensions.serving.compat.langchain import OpenAI

# this ChatOpenAI is imported from mlflow_extensions.serving.compat.langchain
model = ChatOpenAI(
    model="default",  # default is the alias for the model
    base_url="https://<>.com/serving-endpoints/<model-name>",
    api_key="<dapi...>"
)
model.invoke("hello world")
```