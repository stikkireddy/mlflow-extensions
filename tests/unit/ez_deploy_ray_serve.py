import json
from unittest.mock import MagicMock, patch

import pytest

from mlflow_extensions.databricks.deploy.ez_deploy_ray_serve import (
    EzDeployRayServeManager,
    make_base_parameters,
    make_create_json,
    update_cloud_specific_driver_node,
)

DEFAULT_SERVING_NOTEBOOK = (
    "mlflow_extensions/databricks/deploy/ez_deploy_ray_serve_entrypoint"
)

from mlflow_extensions.databricks.deploy.gpu_configs import Cloud

# Mocking EzDeployConfig


class MockEzDeployConfig:
    def serialize_json(self):
        return '{"config_key": "config_value"}'

    class engine_config:
        @staticmethod
        def default_pip_reqs():
            return ["package1==1.0.0", "package2==2.0.0"]

    @property
    def serving_config(self):
        return self

    @property
    def minimum_memory_in_gb(self):
        return 16  # Example value

    @property
    def engine_config(self):
        return self

    def default_pip_reqs(self):
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
    assert (
        create_json["tasks"][0].as_dict()["notebook_task"]["notebook_path"]
        == DEFAULT_SERVING_NOTEBOOK
    )


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


@pytest.fixture
def manager():
    return EzDeployRayServeManager(
        databricks_host="https://test.databricks.com",
        databricks_token="dapiXXXX",
    )


def test_upsert_new_job(manager):
    """
    Test the upsert method when the job does not exist (should create a new job).
    """
    ez_deploy_config = MockEzDeployConfig()

    with patch.object(manager, "client") as mock_client:
        # Simulate that no jobs exist with the given name
        mock_client.jobs.list.return_value = []

        # Mock the jobs.create method
        mock_client.jobs.create.return_value = MagicMock()

        # Patch the make_create_json function to return a known value
        with patch(
            "mlflow_extensions.databricks.deploy.ez_deploy_ray_serve.make_create_json"
        ) as mock_make_create_json:
            mock_create_json = {"name": "test_job"}
            mock_make_create_json.return_value = mock_create_json

            # Call the upsert method
            manager.upsert(
                model_deployment_name="test_model",
                cloud_provider=Cloud.AWS,
                ez_deploy_config=ez_deploy_config,
                hf_secret_scope="test_scope",
                hf_secret_key="test_key",
                entrypoint_git_ref="refs/heads/main",
                min_replica=1,
                max_replica=1,
            )

            # Assert that make_create_json was called with expected parameters
            mock_make_create_json.assert_called_once()

            # Assert that client.jobs.create was called with the correct parameters
            mock_client.jobs.create.assert_called_once_with(**mock_create_json)
