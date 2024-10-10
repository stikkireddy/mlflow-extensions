# Databricks notebook source
dbutils.widgets.text("ez_deploy_config", "")
dbutils.widgets.text("hf_secret_scope", "")
dbutils.widgets.text("hf_secret_key", "")
dbutils.widgets.text("pip_reqs", "")
dbutils.widgets.text("min_replica", "1")
dbutils.widgets.text("max_replica", "1")
dbutils.widgets.text("gpu_config", "")

# COMMAND ----------

ez_deploy_config = dbutils.widgets.get("ez_deploy_config")
hf_secret_scope = dbutils.widgets.get("hf_secret_scope")
hf_secret_key = dbutils.widgets.get("hf_secret_key")
pip_reqs = dbutils.widgets.get("pip_reqs")
min_replica = dbutils.widgets.get("min_replica")
max_replica = dbutils.widgets.get("max_replica")
gpu_config = dbutils.widgets.get("gpu_config")

assert ez_deploy_config, "ez_deploy_config is required"
assert pip_reqs, "pip_reqs is required"
assert gpu_config, "gpu_config is required"

# COMMAND ----------

from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic("pip", f"install {pip_reqs} hf_transfer ray[all]")
ipython.run_line_magic("pip", f"install -U openai")
dbutils.library.restartPython()

# COMMAND ----------

ez_deploy_config = dbutils.widgets.get("ez_deploy_config")
hf_secret_scope = dbutils.widgets.get("hf_secret_scope")
hf_secret_key = dbutils.widgets.get("hf_secret_key")
pip_reqs = dbutils.widgets.get("pip_reqs")
min_replica = dbutils.widgets.get("min_replica")
max_replica = dbutils.widgets.get("max_replica")
gpu_config = dbutils.widgets.get("gpu_config")
assert ez_deploy_config, "ez_deploy_config is required"
assert pip_reqs, "pip_reqs is required"
assert gpu_config, "gpu_config is required"

import json

min_replica = int(min_replica)
max_replica = int(max_replica)
gpu_config = json.loads(gpu_config)

# COMMAND ----------

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
if hf_secret_scope and hf_secret_key:
    os.environ["HF_TOKEN"] = dbutils.secrets.get(
        scope=hf_secret_scope, key=hf_secret_key
    )
os.environ["HF_HOME"] = "/local_disk0/hf_home"

# COMMAND ----------

import logging
from typing import Dict, List, Optional

import ray
from fastapi import FastAPI
from ray import serve
from ray.serve.schema import LoggingConfig
from ray.util.spark import MAX_NUM_WORKER_NODES, setup_ray_cluster, shutdown_ray_cluster
from ray.util.spark.databricks_hook import display_databricks_driver_proxy_url
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from utils import parse_vllm_configs, run_on_every_node
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath, PromptAdapterPath
from vllm.utils import FlexibleArgumentParser

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployConfig
from mlflow_extensions.databricks.deploy.gpu_configs import ALL_VALID_VM_CONFIGS
from mlflow_extensions.databricks.deploy.utils import block_port, unblock_port

# COMMAND ----------

config = EzDeployConfig.from_json(ez_deploy_config)
engine_process = config.to_proc()

# COMMAND ----------

node_info = [
    gpu for gpu in ALL_VALID_VM_CONFIGS if gpu.name == gpu_config["node_type_id"]
]
assert len(node_info) == 1, f"Invalid gpu_config: {gpu_config}"

# block port
shadow_thread, shadow_socket, stop_event = block_port(9989)

if max_replica > 1:

    setup_ray_cluster(
        min_worker_nodes=min_replica,
        max_worker_nodes=max_replica,
        num_cpus_head_node=4,
        num_gpus_worker_node=node_info[0].gpu_count,
        num_cpus_worker_node=node_info[0].cpu_count,
        num_gpus_head_node=0,
    )

    # Pass any custom configuration to ray.init
    ray.init(ignore_reinit_error=True, log_to_driver=False)
else:
    # star local cluster
    ray.init(
        include_dashboard=True,
        ignore_reinit_error=True,
        dashboard_host="0.0.0.0",
        dashboard_port=8888,
        log_to_driver=False,
    )
    display_databricks_driver_proxy_url(sc, 8888, "ray-dashboard")

# COMMAND ----------

from mlflow.pyfunc import PythonModelContext

artifacts = config.download_artifacts()
ctx = PythonModelContext(artifacts=artifacts, model_config={})

# COMMAND ----------


@ray.remote(num_cpus=1)
def download_model():
    config.download_artifacts()


# COMMAND ----------

pg_resources, parsed_args, engine_args = parse_vllm_configs(config, node_info, ctx)

# COMMAND ----------

_ = run_on_every_node(download_model)

# COMMAND ----------

app = FastAPI()


@serve.deployment(
    autoscaling_config={
        "min_replicas": min_replica,
        "max_replicas": max_replica,
        "target_ongoing_requests": 10,
    },
    max_ongoing_requests=20,
    ray_actor_options={"num_cpus": 4},
    logging_config={
        "encoding": "JSON",
        "log_level": "WARN",
        "enable_access_log": False,
    },
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        model_config: EzDeployConfig,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        prompt_adapters: Optional[List[PromptAdapterPath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
    ):
        # self.artifacts = model_config.download_artifacts()
        _ = run_on_every_node(download_model)
        print(f"Starting with engine args: {engine_args}")
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.prompt_adapters = prompt_adapters
        self.request_logger = request_logger
        self.chat_template = chat_template
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            # Determine the name of the served model for the OpenAI client.
            if self.engine_args.served_model_name is not None:
                served_model_names = self.engine_args.served_model_name
            else:
                served_model_names = [self.engine_args.model]
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                served_model_names,
                response_role=self.response_role,
                lora_modules=self.lora_modules,
                prompt_adapters=self.prompt_adapters,
                request_logger=self.request_logger,
                chat_template=self.chat_template,
            )
        print(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


# COMMAND ----------


deploy = VLLMDeployment.options(
    placement_group_bundles=pg_resources, placement_group_strategy="PACK"
).bind(
    engine_args,
    config,
    parsed_args.response_role,
    parsed_args.lora_modules,
    parsed_args.prompt_adapters,
    None,
    parsed_args.chat_template,
)

# COMMAND ----------


unblock_port(shadow_socket, stop_event)
serve.start(http_options={"host": "0.0.0.0", "port": 9989})
serve.run(deploy)

# COMMAND ----------

# serve.shutdown()

# COMMAND ----------

url = f'https://{spark.conf.get("spark.databricks.workspaceUrl")}'
url = url.rstrip("/")
cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
print(
    "Base URL for this endpoint:", f"{url}/driver-proxy-api/o/0/{cluster_id}/9989/v1/"
)

# COMMAND ----------

from openai import OpenAI

# Note: Ray Serve doesn't support all OpenAI client arguments and may ignore some.
client = OpenAI(
    # Replace the URL if deploying your app remotely
    # (e.g., on Anyscale or KubeRay).
    base_url="http://localhost:9989/v1",
    api_key="NOT A REAL KEY",
)
chat_completion = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What are some highly rated restaurants in San Francisco?'",
        },
    ],
    temperature=0.01,
    stream=True,
    max_tokens=100,
)

for chat in chat_completion:
    if chat.choices[0].delta.content is not None:
        print(chat.choices[0].delta.content, end="")

# COMMAND ----------

import time

while True:
    time.sleep(1)

# COMMAND ----------
