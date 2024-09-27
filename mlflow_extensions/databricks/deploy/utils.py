
import ray
from typing import List

from vllm.utils import FlexibleArgumentParser
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser

from mlflow.pyfunc import PythonModelContext

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployConfig

def make_process_and_get_artifacts(config: EzDeployConfig, local_dir=None):
    if local_dir is not None:
        artifacts = config.engine_config.setup_artifacts(local_dir)
    else:
        artifacts = config.engine_config.setup_artifacts()

    engine = config.engine_proc(config=config.engine_config)

    return engine, artifacts



def force_on_node(node_id: str, remote_func_or_actor_class):
    scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node_id, soft=False
    )
    options = {"scheduling_strategy": scheduling_strategy}
    return remote_func_or_actor_class.options(**options)


def run_on_every_node(remote_func_or_actor_class, **remote_kwargs):
    refs = []
    for node in ray.nodes():
        if node["Alive"] and node["Resources"].get("GPU", None):
            refs.append(
                force_on_node(node["NodeID"], remote_func_or_actor_class).remote(
                    **remote_kwargs
                )
            )
    return ray.get(refs)



def parse_vllm_configs(
  config: EzDeployConfig,
  node_info: List,
  ctx:PythonModelContext ):


  vllm_comf = config.engine_config._to_vllm_command(ctx)[3:]
  for index,arg in enumerate(vllm_comf):
    if type(arg) != str:
      vllm_comf[index] = str(arg)
  
  arg_parser = FlexibleArgumentParser(
    description="vLLM OpenAI-Compatible RESTful API server."
)

  parser = make_arg_parser(arg_parser)
  parsed_args = parser.parse_args(args=vllm_comf)
  engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
  engine_args.tensor_parallel_size = node_info[0].gpu_count
  tp = engine_args.tensor_parallel_size
  print(f"Tensor parallelism = {tp}")
  pg_resources = []
  pg_resources.append({"CPU": 4})  # for the deployment replica
  for i in range(tp):
      pg_resources.append({"CPU": 1, 'GPU': 1})  # for the vLLM actors
  return pg_resources ,parsed_args,engine_args