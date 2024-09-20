import functools
from dataclasses import dataclass, field
from typing import List, Optional

import typing_extensions
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import GitSource, JobCluster, JobSettings, Task

from mlflow_extensions.databricks.deploy.gpu_configs import (
    ALL_VALID_VM_CONFIGS,
    Cloud,
    GPUConfig,
)
from mlflow_extensions.version import get_mlflow_extensions_version

if typing_extensions.TYPE_CHECKING:
    from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployConfig

EZ_DEPLOY_LITE_PREFIX = "[EZ_DEPLOY_LITE]"
EZ_DEPLOY_TASK = "deployment"
DEFAULT_RUNTIME = "15.4.x-gpu-ml-scala2.12"
DEFAULT_SERVING_NOTEBOOK = (
    "mlflow_extensions/databricks/deploy/ez_deploy_lite_entrypoint"
)


@dataclass(kw_only=True, frozen=True)
class JobsConfig:
    valid_gpus: List[GPUConfig] = field(default_factory=lambda: ALL_VALID_VM_CONFIGS)
    minimum_memory_in_gb: Optional[int] = None

    def smallest_gpu(self, cloud: Cloud) -> GPUConfig:
        filtered_gpus = [gpu for gpu in self.valid_gpus if gpu.cloud == cloud]
        if self.minimum_memory_in_gb is not None:
            filtered_gpus = [
                gpu
                for gpu in filtered_gpus
                if gpu.total_memory_gb > self.minimum_memory_in_gb
            ]
        if len(filtered_gpus) == 0:
            raise ValueError(
                "No valid gpus in cloud please provide the right gpu config"
            )
        return min(filtered_gpus, key=lambda gpu: gpu.total_memory_gb)


def make_base_parameters(
    config: "EzDeployConfig", hf_secret_scope: str, hf_secret_key: str
):
    return {
        "ez_deploy_config": config.serialize_json(),
        "hf_secret_scope": hf_secret_scope or "",
        "hf_secret_key": hf_secret_key or "",
        "pip_reqs": " ".join(config.engine_config.default_pip_reqs()),
    }


def make_cloud_specific_attrs(cloud: Cloud):
    if cloud == Cloud.GCP:
        return {
            "gcp_attributes": {
                "use_preemptible_executors": False,
                "availability": "ON_DEMAND_GCP",
                "zone_id": "HA",
            }
        }
    if cloud == Cloud.AWS:
        return {
            "aws_attributes": {
                "first_on_demand": 1,
                "availability": "SPOT_WITH_FALLBACK",
                "zone_id": "auto",
                "instance_profile_arn": None,
                "spot_bid_price_percent": 100,
            },
        }
    if cloud == Cloud.AZURE:
        return {
            "azure_attributes": {
                "first_on_demand": 1,
                "availability": "ON_DEMAND_AZURE",
                "spot_bid_max_price": -1,
            }
        }
    raise ValueError(f"Cloud {cloud} is not supported")


def make_create_json(
    *,
    job_name: str,
    minimum_memory_in_gb: int,
    cloud_provider: Cloud,
    ez_deploy_config: "EzDeployConfig",
    huggingface_secret_scope: str,
    huggingface_secret_key: str,
    task_name: str = EZ_DEPLOY_TASK,
    runtime: str = DEFAULT_RUNTIME,
    notebook_path: str = DEFAULT_SERVING_NOTEBOOK,
    specific_git_ref: str = None,
):
    vm = JobsConfig(minimum_memory_in_gb=minimum_memory_in_gb).smallest_gpu(
        cloud_provider
    )
    gpu_node = {
        "spark_version": runtime,
        "spark_conf": {
            "spark.master": "local[*, 4]",
            "spark.databricks.cluster.profile": "singleNode",
        },
        "node_type_id": vm.name,
        "driver_node_type_id": vm.name,
        "custom_tags": {"ResourceClass": "SingleNode"},
        "enable_elastic_disk": True,
        "data_security_mode": "NONE",
        "runtime_engine": "STANDARD",
        "num_workers": 0,
    }
    gpu_node.update(make_cloud_specific_attrs(cloud_provider))

    return {
        "name": job_name,
        "timeout_seconds": 0,
        "max_concurrent_runs": 1,
        "tasks": [
            Task.from_dict(
                {
                    "task_key": task_name,
                    "run_if": "ALL_SUCCESS",
                    "notebook_task": {
                        "notebook_path": notebook_path,
                        "source": "GIT",
                        "base_parameters": make_base_parameters(
                            ez_deploy_config,
                            huggingface_secret_scope,
                            huggingface_secret_key,
                        ),
                    },
                    "job_cluster_key": "deployment_gpu",
                    "timeout_seconds": 0,
                }
            )
        ],
        "git_source": GitSource.from_dict(
            {
                "git_url": "https://github.com/stikkireddy/mlflow-extensions.git",
                "git_provider": "gitHub",
                "git_branch": specific_git_ref or get_mlflow_extensions_version(),
            }
        ),
        "job_clusters": [
            JobCluster.from_dict(
                {"job_cluster_key": "deployment_gpu", "new_cluster": gpu_node}
            )
        ],
    }


class EzDeployLiteManager:

    def __init__(
        self,
        databricks_host: str = None,
        databricks_token: str = None,
        prefix: str = EZ_DEPLOY_LITE_PREFIX,
    ):
        self._prefix = prefix
        self.databricks_host = databricks_host
        self.databricks_token = databricks_token
        self.client = WorkspaceClient(host=databricks_host, token=databricks_token)

    def make_name(self, name):
        return f"{self._prefix} {name}"

    @functools.lru_cache(maxsize=32)
    def get_jobs(self, job_name):
        return [job for job in self.client.jobs.list(name=job_name)]

    def exists(self, model_deployment_name):
        job_name = self.make_name(model_deployment_name)
        return len(self.get_jobs(job_name)) >= 1

    def make_oai_url(self, cluster_id):
        return f"{self.client.config.host.rstrip('/')}/driver-proxy-api/o/0/{cluster_id}/9989/v1/"

    def get_openai_url(
        self, model_deployment_name, deployment_name: str = EZ_DEPLOY_TASK
    ):
        job_name = self.make_name(model_deployment_name)
        if not self.exists(model_deployment_name):
            raise ValueError(f"Model deployment {model_deployment_name} does not exist")
        job = [job for job in self.client.jobs.list(name=job_name)][0]
        job_id = job.job_id
        for run in self.client.jobs.list_runs(
            active_only=True, expand_tasks=True, job_id=job_id
        ):
            for task in run.tasks:
                if task.task_key == deployment_name:
                    if task.cluster_instance is None:
                        raise ValueError(
                            f"Model deployment {model_deployment_name} has no cluster"
                        )
                    if task.cluster_instance.cluster_id is None:
                        raise ValueError(
                            f"Model deployment {model_deployment_name} has no cluster id"
                        )
                    return self.make_oai_url(task.cluster_instance.cluster_id)
        raise ValueError(
            f"Model deployment {model_deployment_name} has no running cluster endpoint"
        )

    def upsert(
        self,
        model_deployment_name,
        cloud_provider,
        ez_deploy_config: EzDeployConfig,
        hf_secret_scope=None,
        hf_secret_key=None,
        entrypoint_git_ref: str = None,
    ):
        job_name = self.make_name(model_deployment_name)
        assert (
            ez_deploy_config.serving_config.minimum_memory_in_gb
        ), "Minimum memory in GB is not set"
        create_json = make_create_json(
            job_name=job_name,
            minimum_memory_in_gb=ez_deploy_config.serving_config.minimum_memory_in_gb,
            cloud_provider=cloud_provider,
            ez_deploy_config=ez_deploy_config,
            huggingface_secret_scope=hf_secret_scope,
            huggingface_secret_key=hf_secret_key,
            specific_git_ref=entrypoint_git_ref,
        )
        if self.exists(model_deployment_name):
            job_id = self.get_jobs(job_name)[0].job_id
            reset_data = {"new_settings": JobSettings(**create_json)}
            return self.client.jobs.reset(job_id=job_id, **reset_data)
        self.client.jobs.create(**create_json)

    def start_server(self, model_deployment_name):
        job_name = self.make_name(model_deployment_name)
        if not self.exists(model_deployment_name):
            raise ValueError(f"Model deployment {model_deployment_name} does not exist")
        job = list(self.client.jobs.list(name=job_name))[0]
        job_id = job.job_id
        if len(list(self.client.jobs.list_runs(active_only=True, job_id=job_id))) == 0:
            return self.client.jobs.run_now(job_id=job_id)
        raise ValueError(
            f"Model deployment {model_deployment_name} has an active run, please cancel the run "
            f"and start a new run."
        )
