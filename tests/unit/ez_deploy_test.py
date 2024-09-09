import os
import random
import re
import shutil
import string
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Sequence, Tuple

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import ModelVersionInfo
from mlflow import MlflowClient

import mlflow_extensions.databricks.prebuilt as prebuilt
from mlflow_extensions.databricks.deploy.ez_deploy import (
    EzDeploy,
    EzDeployConfig,
    ServingConfig,
)

ez_deploy_configs: List[EzDeployConfig] = [
    prebuilt.vision.vllm.PHI_3_5_VISION_INSTRUCT_12K_CONFIG,
]


def is_directory_empty(path: Path) -> bool:
    is_empty: bool = not any(path.iterdir())
    return is_empty


@pytest.fixture
def database_name(workspace_client: WorkspaceClient) -> Generator[str, None, None]:
    catalog_name: str = "main"

    dt_string: datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    random_suffix: str = re.sub(r"[^A-Za-z0-9]", "", dt_string)

    database_name: str = f"mlflow_extensions_test_{random_suffix}"

    if not any([c.name == catalog_name for c in workspace_client.catalogs.list()]):
        workspace_client.catalogs.create(name=catalog_name)

    if not any(
        [
            s.name == database_name
            for s in workspace_client.schemas.list(catalog_name=catalog_name)
        ]
    ):
        workspace_client.schemas.create(catalog_name=catalog_name, name=database_name)

    # Yield to the test
    full_name: str = f"{catalog_name}.{database_name}"
    yield full_name

    workspace_client.schemas.delete(full_name=full_name, force=True)


@pytest.fixture
def config() -> EzDeployConfig:
    config: EzDeployConfig = EzDeployConfig(
        name="test",
        engine_config=None,
        engine_proc=None,
        serving_config=ServingConfig(),
    )
    return config


@pytest.fixture
def deploy(config: EzDeployConfig) -> EzDeploy:
    return EzDeploy(config=config, registered_model_name="catalog.schema.model")


@pytest.mark.unit
def test_should_init(config: EzDeployConfig) -> None:
    _ = EzDeploy(config=config, registered_model_name="catalog.schema.model")


@pytest.mark.unit
def test_init_should_raise_when_registered_model_name_is_not_fully_qualified(
    config: EzDeployConfig,
) -> None:
    with pytest.raises(AssertionError) as ae:
        _ = EzDeploy(config=config, registered_model_name="unqualified_model_name")


@pytest.mark.integration
@pytest.mark.parametrize("config", ez_deploy_configs)
def test_should_download_provided_path(
    config: EzDeployConfig, tmp_path_factory, database_name: str
) -> None:

    registered_model_name: str = f"{database_name}.phi_3_5_vision_instruct_12k"

    deploy: EzDeploy = EzDeploy(
        config=config,
        registered_model_name=registered_model_name,
    )

    dir_name: str = "models"
    download_path: Path = tmp_path_factory.mktemp(dir_name)
    try:
        deploy.download(local_dir=download_path)
        assert deploy._downloaded and not is_directory_empty(download_path)
    finally:
        shutil.rmtree(download_path)


@pytest.mark.unit
def test_should_deploy(deploy: EzDeploy) -> None:
    assert deploy is not None


@pytest.mark.unit
@pytest.mark.parametrize("config", ez_deploy_configs)
def test_register_should_raise_when_not_downloaded(
    deploy: EzDeploy, database_name: str
) -> None:

    registered_model_name: str = f"{database_name}.phi_3_5_vision_instruct_12k"

    deploy: EzDeploy = EzDeploy(
        config=config,
        registered_model_name=registered_model_name,
    )
    with pytest.raises(AssertionError) as ae:
        deploy.register()


@pytest.mark.skipif(
    "DATABRICKS_HOST" not in os.environ,
    reason="Missing environment variable: DATABRICKS_HOST",
)
@pytest.mark.skipif(
    "DATABRICKS_TOKEN" not in os.environ,
    reason="Missing environment variable: DATABRICKS_TOKEN",
)
@pytest.mark.integration
@pytest.mark.databricks
@pytest.mark.parametrize("config", ez_deploy_configs)
def test_register(
    config: EzDeployConfig, workspace_client: WorkspaceClient, database_name: str
) -> None:

    registered_model_name: str = f"{database_name}.phi_3_5_vision_instruct_12k"

    deploy: EzDeploy = EzDeploy(
        config=config,
        registered_model_name=registered_model_name,
    )

    deploy.download()
    deploy.register()

    models: List[ModelVersionInfo] = list(
        workspace_client.model_versions.list(full_name=registered_model_name)
    )
    assert len(models) > 0

    workspace_client.model_versions.delete(
        full_name=registered_model_name, version=models[0].version
    )
