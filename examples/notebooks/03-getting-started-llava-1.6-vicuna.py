# Databricks notebook source
# MAGIC %pip install mlflow-extensions
# MAGIC %pip install mlflow -U
# MAGIC %pip install sglang==0.2.13 outlines==0.0.44
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeploy
from mlflow_extensions.databricks.prebuilt import prebuilt

deployer = EzDeploy(
  config=prebuilt.vision.sglang.LLAVA_NEXT_LLAMA3_8B_CONFIG,
  registered_model_name="main.default.llava_next_llama3_8b_based"
)

deployer.download()

deployer.register()

endpoint_name = "llava_next_llama3_8b_sglang"

deployer.deploy(endpoint_name)

# COMMAND ----------

from mlflow_extensions.serving.compat.openai import OpenAI
from mlflow.utils.databricks_utils import get_databricks_host_creds

workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
endpoint_name = "llava_next_llama3_8b_sglang"
endpoint_url = f"https://{workspace_host}/serving-endpoints/{endpoint_name}/invocations"

token = get_databricks_host_creds().token

client = OpenAI(
  base_url=endpoint_url,
  api_key=token
)
client = OpenAI(base_url=endpoint_url, api_key=token)
# print(client.models.list())
response = client.chat.completions.create(
  model="lmms-lab/llama3-llava-next-8b",
  max_tokens=256,
  messages=[
    {"role": "user", "content": [
                {"type": "text", "text": "Explain the content of the image? What is in the background?"},
                {
                    "type": "image_url",
                    "image_url": {
                      "url": "https://richmedia.ca-richimage.com/ImageDelivery/imageService?profileId=12026540&id=1859027&recipeId=728"
                    },
                },
       ],
     }
  ],
)
print(response.choices[0].message.content)

# COMMAND ----------

from mlflow_extensions.serving.compat.sglang import RuntimeEndpoint
from mlflow.utils.databricks_utils import get_databricks_host_creds
from sglang import set_default_backend
from sglang.srt.constrained import build_regex_from_object
import sglang as sgl
from pydantic import BaseModel
from typing import Literal

import requests

# the first run takes a bit longer to compile the FSM on the server, all subsequent requests will be fast
# the first call may take 10-30 seconds depending on the complexity of the pydantic object

token = get_databricks_host_creds().token

# connect sglang frontend (this python code) to the backend (model serving endpoint)
set_default_backend(RuntimeEndpoint(endpoint_url, token))


class Fashion(BaseModel):
    color: Literal["black", "blue", "gray"]
    material: Literal["silk", "denim", "fabric"]
    gender: Literal["male", "female"]

fashion = build_regex_from_object(Fashion)
# fix a small regex bug with outlines + sglang for strings
fashion = fashion.replace(r"""([^"\\\x00-\x1f\x7f-\x9f]|\\\\)""", "[\w\d\s]")
print(fashion)

@sgl.function
def image_qa(s, image_file):
    s += sgl.user(sgl.image(image_file))
    s += "Fill in the details about the item... \n"
    s += sgl.gen(
        "clothing_details",
        max_tokens=128,
        temperature=0,
        regex=fashion,  # Requires pydantic >= 2.0
    )


# URL you want to fetch
url = "https://richmedia.ca-richimage.com/ImageDelivery/imageService?profileId=12026540&id=1859027&recipeId=728"

response = requests.get(url)
response.raise_for_status()  # Check for request errors

# only need to send the bytes no download, etc
data = image_qa.run(
  image_file=response.content,
)

# access by the generation key you asked for
print(data["clothing_details"])



# COMMAND ----------

from mlflow_extensions.serving.compat.sglang import RuntimeEndpoint
from mlflow.utils.databricks_utils import get_databricks_host_creds
from sglang import set_default_backend
from sglang.srt.constrained import build_regex_from_object
import sglang as sgl
from pydantic import BaseModel
from typing import Literal

import requests

# the first run takes a bit longer to compile the FSM on the server, all subsequent requests will be fast
# the first call may take 10-30 seconds depending on the complexity of the pydantic object

token = get_databricks_host_creds().token

# connect sglang frontend (this python code) to the backend (model serving endpoint)
set_default_backend(RuntimeEndpoint(endpoint_url, token))


class FashionProblems(BaseModel):
    description: str
    clothing_type: Literal["shirt", "pants", "dress", "skirt", "shoes"]
    color: Literal["black", "blue", "gray"]
    material: Literal["silk", "denim", "fabric"]

fashion_problems = build_regex_from_object(FashionProblems)
# fix a small regex bug with outlines + sglang for strings
fashion_problems = fashion_problems.replace(r"""([^"\\\x00-\x1f\x7f-\x9f]|\\\\)""", "[\w\d\s]")
print(fashion_problems)

@sgl.function
def image_qa(s, image_file):
    s += sgl.user(sgl.image(image_file))
    s += "Fill in the problems about the product... \n"
    s += sgl.gen(
        "clothing_details",
        max_tokens=128,
        temperature=0,
        regex=fashion_problems,  # Requires pydantic >= 2.0
    )


# URL you want to fetch
url = "https://m.media-amazon.com/images/I/51a94AxNRPL.jpg"

response = requests.get(url)
response.raise_for_status()  # Check for request errors

# only need to send the bytes no download, etc
data = image_qa.run(
  image_file=response.content,
)

# access by the generation key you asked for
print(data["clothing_details"])



# COMMAND ----------

from mlflow_extensions.serving.compat.sglang import RuntimeEndpoint
from mlflow.utils.databricks_utils import get_databricks_host_creds
from sglang import set_default_backend
from sglang.srt.constrained import build_regex_from_object
import sglang as sgl
from pydantic import BaseModel
from typing import Literal, List

import requests

# the first run takes a bit longer to compile the FSM on the server, all subsequent requests will be fast
# the first call may take 10-30 seconds depending on the complexity of the pydantic object

token = get_databricks_host_creds().token

# connect sglang frontend (this python code) to the backend (model serving endpoint)
set_default_backend(RuntimeEndpoint(endpoint_url, token))


class StockoutProblems(BaseModel):
    description: str
    stockout: bool
    types_of_products: List[Literal["food", "clothing", "appliances", "durables"]]


stockout_problems = build_regex_from_object(StockoutProblems)
# fix a small regex bug with outlines + sglang for strings
stockout_problems = stockout_problems.replace(r"""([^"\\\x00-\x1f\x7f-\x9f]|\\\\)""", "[\w\d\s]")
print(stockout_problems)

@sgl.function
def image_qa(s, image_file):
    s += sgl.user(sgl.image(image_file))
    s += "Is there an out of stock situation? What type of product seems to be out of stock? (food/clothing/appliances/durables) \n"
    s += sgl.gen(
        "stockout_details",
        max_tokens=128,
        temperature=0,
        regex=stockout_problems,  # Requires pydantic >= 2.0
    )


# URL you want to fetch
url = "https://assets.eposnow.com/public/content-images/pexels-roy-broo-empty-shelves-grocery-items.jpg"

response = requests.get(url)
response.raise_for_status()  # Check for request errors

# only need to send the bytes no download, etc
data = image_qa.run(
  image_file=response.content,
)

# access by the generation key you asked for
print(data["stockout_details"])



# COMMAND ----------
