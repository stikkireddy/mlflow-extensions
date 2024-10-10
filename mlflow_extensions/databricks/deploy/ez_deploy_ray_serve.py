import json
from typing import Dict

import typing_extensions
from databricks.sdk.service.jobs import GitSource, JobCluster, JobSettings, Task

from mlflow_extensions.databricks.deploy.ez_deploy_lite import (
    DEFAULT_RUNTIME,
    EZ_DEPLOY_TASK,
    EzDeployLiteManager,
    JobsConfig,
    make_cloud_specific_attrs,
)
from mlflow_extensions.databricks.deploy.gpu_configs import Cloud
from mlflow_extensions.version import get_mlflow_extensions_version

if typing_extensions.TYPE_CHECKING:
    from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployConfig


EZ_DEPLOY_LITE_PREFIX = "[EZ_DEPLOY_RAY_SERVE]"
DEFAULT_SERVING_NOTEBOOK = (
    "mlflow_extensions/databricks/deploy/ez_deploy_ray_serve_entrypoint"
)


def make_base_parameters(
    config: "EzDeployConfig",
    hf_secret_scope: str,
    hf_secret_key: str,
    min_replica: int,
    max_replica: int,
    gpu_node=Dict,
):
    """
    Function to Format the Parameter Arguements given to worflow to string
    """
    return {
        "ez_deploy_config": config.serialize_json(),
        "hf_secret_scope": hf_secret_scope or "",
        "hf_secret_key": hf_secret_key or "",
        "min_replica": min_replica,
        "max_replica": max_replica,
        "gpu_config": json.dumps(gpu_node),
        "pip_reqs": " ".join(config.engine_config.default_pip_reqs()),
    }


def update_cloud_specific_driver_node(cloud: Cloud):
    if cloud == Cloud.GCP:
        return {"driver_node_type_id": "n2-highmem-4"}
    if cloud == Cloud.AWS:
        return {"driver_node_type_id": "i3.xlarge"}
    if cloud == Cloud.AZURE:
        return {"driver_node_type_id": "Standard_DS3_v2"}
    raise ValueError(f"Cloud {cloud} is not supported")


def make_create_json(
    *,
    job_name: str,
    minimum_memory_in_gb: int,
    min_replica: int,
    max_replica: int,
    cloud_provider: Cloud,
    ez_deploy_config: "EzDeployConfig",
    huggingface_secret_scope: str,
    huggingface_secret_key: str,
    task_name: str = EZ_DEPLOY_TASK,
    runtime: str = DEFAULT_RUNTIME,
    notebook_path: str = DEFAULT_SERVING_NOTEBOOK,
    specific_git_ref: str = None,
    git_url: str = "https://github.com/stikkireddy/mlflow-extensions.git",
):
    vm = JobsConfig(minimum_memory_in_gb=minimum_memory_in_gb).smallest_gpu(
        cloud_provider
    )
    if max_replica == 1:
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
    else:
        gpu_node = {
            "spark_version": runtime,
            "node_type_id": vm.name,
            "driver_node_type_id": vm.name,
            "enable_elastic_disk": True,
            "data_security_mode": "NONE",
            "runtime_engine": "STANDARD",
        }
        if min_replica == max_replica:
            gpu_node["num_workers"] = min_replica
        else:
            gpu_node["autoscale"] = {
                "min_workers": min_replica,
                "max_workers": max_replica,
            }
        gpu_node.update(update_cloud_specific_driver_node(cloud_provider))
    gpu_node.update(make_cloud_specific_attrs(cloud_provider))

    if specific_git_ref:
        parts = specific_git_ref.split("/", maxsplit=1)
        git_type = "git_branch"
        if parts[0].startswith("commit"):
            git_type = "git_commit"
        elif parts[0].startswith("tag"):
            git_type = "git_tag"

        git_source = GitSource.from_dict(
            {
                "git_url": git_url,
                "git_provider": "gitHub",
                git_type: parts[1],
            }
        )
    else:
        git_source = GitSource.from_dict(
            {
                "git_url": git_url,
                "git_provider": "gitHub",
                "git_tag": f"v{get_mlflow_extensions_version()}",
            }
        )

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
                            min_replica,
                            max_replica,
                            gpu_node,
                        ),
                    },
                    "job_cluster_key": "deployment_gpu",
                    "timeout_seconds": 0,
                }
            )
        ],
        "git_source": git_source,
        "job_clusters": [
            JobCluster.from_dict(
                {"job_cluster_key": "deployment_gpu", "new_cluster": gpu_node}
            )
        ],
    }


class EzDeployRayServeManager(EzDeployLiteManager):

    def __init__(
        self,
        databricks_host: str = None,
        databricks_token: str = None,
        prefix: str = EZ_DEPLOY_LITE_PREFIX,
    ):
        super().__init__(databricks_host, databricks_token, prefix)

    def upsert(
        self,
        model_deployment_name,
        cloud_provider,
        ez_deploy_config: "EzDeployConfig",
        hf_secret_scope=None,
        hf_secret_key=None,
        entrypoint_git_ref: str = None,
        entrypoint_git_url: str = "https://github.com/stikkireddy/mlflow-extensions.git",
        min_replica: int = 1,
        max_replica: int = 1,
    ):
        job_name = self.make_name(model_deployment_name)
        assert (
            ez_deploy_config.serving_config.minimum_memory_in_gb
        ), "Minimum memory in GB is not set"
        create_json = make_create_json(
            job_name=job_name,
            minimum_memory_in_gb=ez_deploy_config.serving_config.minimum_memory_in_gb,
            min_replica=min_replica,
            max_replica=max_replica,
            cloud_provider=cloud_provider,
            ez_deploy_config=ez_deploy_config,
            huggingface_secret_scope=hf_secret_scope,
            huggingface_secret_key=hf_secret_key,
            specific_git_ref=entrypoint_git_ref,
            git_url=entrypoint_git_url,
        )
        if self.exists(model_deployment_name):
            job_id = self.get_jobs(job_name)[0].job_id
            reset_data = {"new_settings": JobSettings(**create_json)}
            return self.client.jobs.reset(job_id=job_id, **reset_data)
        self.client.jobs.create(**create_json)
