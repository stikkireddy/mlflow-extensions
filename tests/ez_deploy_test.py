from typing import List, Sequence, Tuple

import pytest

import mlflow_extensions.databricks.prebuilt as prebuilt
from mlflow_extensions.databricks.deploy.ez_deploy import (
    EzDeploy,
    EzDeployConfig,
    ServingConfig,
)


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
def test_should_raise_when_registered_model_name_is_not_fully_qualified(
    config: EzDeployConfig,
) -> None:
    with pytest.raises(AssertionError) as ae:
        _ = EzDeploy(config=config, registered_model_name="unqualified_model_name")


ez_deploy_configs: List[EzDeployConfig] = [
    prebuilt.vision.vllm.PHI_3_5_VISION_INSTRUCT_12K_CONFIG,
]


@pytest.mark.integration
@pytest.mark.parametrize("config", ez_deploy_configs)
def test_should_download(config: EzDeployConfig) -> None:
    print(config)
    deploy: EzDeploy = EzDeploy(
        config=config,
        registered_model_name="main.default.sri_phi_3_5_vision_instruct_12k",
    )

    deploy.download()


@pytest.mark.unit
def test_should_deploy(deploy: EzDeploy) -> None:
    assert deploy is not None

@pytest.mark.unit
def test_should_register(deploy: EzDeploy) -> None:
    assert deploy is not None
