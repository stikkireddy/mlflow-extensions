import time
from dataclasses import dataclass, field
from typing import List, Optional,Dict

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

from mlflow_extensions.databricks.deploy.ez_deploy_lite import (
EZ_DEPLOY_TASK ,
DEFAULT_RUNTIME,
JobsConfig,
make_cloud_specific_attrs,
EzDeployLiteManager
)

EZ_DEPLOY_LITE_PREFIX = "[EZ_DEPLOY_RAY_SERVE]"
DEFAULT_SERVING_NOTEBOOK = (
    "mlflow_extensions/databricks/deploy/ez_deploy_lite_ray_serve"
)

def make_base_parameters(
    config: "EzDeployConfig", hf_secret_scope: str, hf_secret_key: str,
    Replica: int,gpu_node = Dict
):
    return {
        "ez_deploy_config": config.serialize_json(),
        "hf_secret_scope": hf_secret_scope or "",
        "hf_secret_key": hf_secret_key or "",
        "replica" : Replica or 1,
        "gpu_config" : gpu_node,
        "pip_reqs": " ".join(config.engine_config.default_pip_reqs()),
    }

def update_cloud_specific_driver_node(cloud: Cloud):
    if cloud == Cloud.GCP:
        return {
            "driver_node_type_id": "n2-highmem-4"
        }
    if cloud == Cloud.AWS:
        return {
            "driver_node_type_id" :"i3.xlarge"
            }
    if cloud == Cloud.AZURE:
        return {
            "driver_node_type_id": "Standard_DS3_v2"
            }
    raise ValueError(f"Cloud {cloud} is not supported")


def make_create_json(
    *,
    job_name: str,
    minimum_memory_in_gb: int,
    Replica,
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
    if Replica == 1:
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
            "custom_tags": {"ResourceClass": "SingleNode"},
            "enable_elastic_disk": True,
            "data_security_mode": "NONE",
            "runtime_engine": "STANDARD",
            "num_workers": Replica,
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
                "git_url": "https://github.com/puneet-jain159/mlflow-extensions.git",
                "git_provider": "gitHub",
                git_type: parts[1],
            }
        )
    else:
        git_source = GitSource.from_dict(
            {
                "git_url": "https://github.com/puneet-jain159/mlflow-extensions.git",
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
                            Replica,
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
        super().__init__(databricks_host,
                         databricks_token,
                         prefix) 


    def upsert(
        self,
        model_deployment_name,
        Replica,
        cloud_provider,
        ez_deploy_config: "EzDeployConfig",
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
            Replica = Replica,
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
        active_runs = list(self.client.jobs.list_runs(active_only=True, job_id=job_id))
        if len(active_runs) == 0:
            self.client.jobs.run_now(job_id=job_id)
            time.sleep(5)
            active_runs = list(
                self.client.jobs.list_runs(active_only=True, job_id=job_id)
            )
            print("Running model at: ", active_runs[0].run_page_url)
            return

        active_runs_urls = [run.run_page_url for run in active_runs]
        raise ValueError(
            f"Model deployment {model_deployment_name} has an active run, please cancel the run "
            f"and start a new run. Please cancel these: {str(active_runs_urls)}"
        )