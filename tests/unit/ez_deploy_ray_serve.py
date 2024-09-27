import pytest
from unittest.mock import MagicMock
import json

from mlflow_extensions.databricks.deploy.ez_deploy_ray_serve import (
    update_cloud_specific_driver_node,
    make_base_parameters,
    make_create_json,
)

DEFAULT_SERVING_NOTEBOOK = 'mlflow_extensions/databricks/deploy/ez_deploy_ray_serve_entrypoint'

from mlflow_extensions.databricks.deploy.gpu_configs import Cloud

# Mocking EzDeployConfig


class MockEzDeployConfig:
    def serialize_json(self):
        return '{"config_key": "config_value"}'

    class engine_config:
        @staticmethod
        def default_pip_reqs():
            return ["package1==1.0.0", "package2==2.0.0"]


def test_make_base_parameters():
    config = MockEzDeployConfig()
    hf_secret_scope = "test_scope"
    hf_secret_key = "test_key"
    min_replica = 1
    max_replica = 2
    gpu_node = {"gpu_key": "gpu_value"}

    expected_output = {
        "ez_deploy_config": config.serialize_json(),
        "hf_secret_scope": hf_secret_scope,
        "hf_secret_key": hf_secret_key,
        "min_replica": min_replica,
        "max_replica": max_replica,
        "gpu_config": json.dumps(gpu_node),
        "pip_reqs": " ".join(config.engine_config.default_pip_reqs()),
    }

    result = make_base_parameters(
        config, hf_secret_scope, hf_secret_key, min_replica, max_replica, gpu_node
    )
    assert result == expected_output


def test_update_cloud_specific_driver_node_gcp():

    expected_output = {"driver_node_type_id": "n2-highmem-4"}
    result = update_cloud_specific_driver_node(Cloud.GCP)
    assert result == expected_output


def test_make_create_json():

    job_name = "test_job"
    minimum_memory_in_gb = 16
    min_replica = 1
    max_replica = 1
    cloud_provider = Cloud.GCP
    ez_deploy_config = MockEzDeployConfig()
    huggingface_secret_scope = "hf_scope"
    huggingface_secret_key = "hf_key"
    specific_git_ref = None

    create_json = make_create_json(
        job_name=job_name,
        minimum_memory_in_gb=minimum_memory_in_gb,
        min_replica=min_replica,
        max_replica=max_replica,
        cloud_provider=cloud_provider,
        ez_deploy_config=ez_deploy_config,
        huggingface_secret_scope=huggingface_secret_scope,
        huggingface_secret_key=huggingface_secret_key,
        specific_git_ref=specific_git_ref,
    )

    # Since create_json is complex, we can check for presence of some keys
    assert create_json["name"] == job_name
    assert "tasks" in create_json
    assert "job_clusters" in create_json
    assert create_json["tasks"][0].as_dict()["notebook_task"]['notebook_path'] == DEFAULT_SERVING_NOTEBOOK


def test_make_create_json_with_autoscale():

    job_name = "test_job"
    minimum_memory_in_gb = 16
    min_replica = 1
    max_replica = 2
    cloud_provider = Cloud.AWS
    ez_deploy_config = MockEzDeployConfig()
    huggingface_secret_scope = "hf_scope"
    huggingface_secret_key = "hf_key"
    specific_git_ref = None

    create_json = make_create_json(
        job_name=job_name,
        minimum_memory_in_gb=minimum_memory_in_gb,
        min_replica=min_replica,
        max_replica=max_replica,
        cloud_provider=cloud_provider,
        ez_deploy_config=ez_deploy_config,
        huggingface_secret_scope=huggingface_secret_scope,
        huggingface_secret_key=huggingface_secret_key,
        specific_git_ref=specific_git_ref,
    )

    # Check that autoscale is set
    print(create_json["job_clusters"][0].as_dict())
    cluster_config = create_json["job_clusters"][0].as_dict()["new_cluster"]
    assert "autoscale" in cluster_config
    assert cluster_config["autoscale"]["min_workers"] == min_replica
    assert cluster_config["autoscale"]["max_workers"] == max_replica
