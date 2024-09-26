# Databricks notebook source
dbutils.widgets.text("ez_deploy_config", '{"name": "qwen2_vl_7b_instruct", "engine_config": {"model": "Qwen/Qwen2-VL-7B-Instruct", "host": "0.0.0.0", "port": 9989, "openai_api_path": "v1", "ensure_supported_models": true, "library_overrides": {"transformers": "git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830", "accelerate": "accelerate==0.31.0", "vllm": "vllm==0.6.1"}, "entrypoint_module": "vllm.entrypoints.openai.api_server", "enable_experimental_chunked_prefill": false, "max_num_batched_tokens": null, "enable_prefix_caching": false, "vllm_command_flags": {"--gpu-memory-utilization": 0.98, "--distributed-executor-backend": "ray"}, "trust_remote_code": false, "max_model_len": null, "served_model_alias": null, "guided_decoding_backend": "outlines", "tokenizer": null, "max_num_images": 5, "max_num_videos": null, "max_num_audios": null, "model_artifact_key": "model", "verify_chat_template": true, "tokenizer_artifact_key": "tokenizer", "tokenizer_config_file": "tokenizer_config.json", "chat_template_key": "chat_template", "tokenizer_mode": null}, "engine_proc": "VLLMEngineProcess", "pip_config_override": null}')
dbutils.widgets.text("hf_secret_scope", "")
dbutils.widgets.text("hf_secret_key", "")
dbutils.widgets.text("pip_reqs", "httpx==0.27.0 psutil==6.0.0 filelock==3.15.4 mlflow==2.16.0 mlflow-extensions vllm==0.6.1 outlines==0.0.46 git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate==0.31.0")

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
ipython.run_line_magic("pip", f"install {pip_reqs} hf_transfer ray[all]")
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

from typing import Dict, Optional, List
import logging

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve,init

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath, PromptAdapterPath
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.logger import RequestLogger

# COMMAND ----------

init(include_dashboard=True ,ignore_reinit_error=True, dashboard_host = "0.0.0.0",dashboard_port= 8888)

# COMMAND ----------

from ray.util.spark.databricks_hook import display_databricks_driver_proxy_url

display_databricks_driver_proxy_url(sc,8888, "ray-dashboard")

# COMMAND ----------

from mlflow.pyfunc import PythonModelContext

ctx = PythonModelContext(artifacts=artifacts, model_config={})

# COMMAND ----------

vllm_comf = config.engine_config._to_vllm_command(ctx)[3:]
for index,arg in enumerate(vllm_comf):
  if type(arg) != str:
    vllm_comf[index] = str(arg)

# COMMAND ----------

arg_parser = FlexibleArgumentParser(
    description="vLLM OpenAI-Compatible RESTful API server."
)

parser = make_arg_parser(arg_parser)
parsed_args = parser.parse_args(args=vllm_comf)
engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
engine_args.worker_use_ray = True

tp = engine_args.tensor_parallel_size
print(f"Tensor parallelism = {tp}")
pg_resources = []
pg_resources.append({"CPU": 1})  # for the deployment replica
for i in range(tp):
    pg_resources.append({"CPU": 1, 'GPU': 1})  # for the vLLM actors


# COMMAND ----------

app = FastAPI()


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 2,
        "target_ongoing_requests": 5,
    },
    max_ongoing_requests=10,
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        prompt_adapters: Optional[List[PromptAdapterPath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
    ):
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
                response_role =self.response_role,
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


deploy =  VLLMDeployment.options(
        placement_group_bundles=pg_resources, placement_group_strategy="STRICT_PACK"
    ).bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.prompt_adapters,
        None,
        parsed_args.chat_template,

    )

# COMMAND ----------

serve.start(http_options = {'host' : "0.0.0.0", 'port' : 9989})
serve.run(deploy)

# COMMAND ----------

# serve.shutdown()

# COMMAND ----------

# import ray

# from mlflow_extensions.testing.helper import kill_processes_containing

# ray.shutdown()
# kill_processes_containing("vllm")
# kill_processes_containing("ray")
# kill_processes_containing("sglang")
# kill_processes_containing("from multiprocessing")
# engine_process.start_proc(ctx, health_check_thread=False)

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



# COMMAND ----------

import time

while True:
    time.sleep(1)
