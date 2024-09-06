from mlflow_extensions.databricks.deploy.ez_deploy import EzDeploy, EzDeployConfig, ServingConfig

import pytest

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

def test_should_init(config: EzDeployConfig) -> None:
    _ = EzDeploy(config=config, registered_model_name="catalog.schema.model")
        
def test_should_raise_when_registered_model_name_is_not_fully_qualified(config: EzDeployConfig) -> None:
    with pytest.raises(AssertionError) as ae:
        _ = EzDeploy(config=config, registered_model_name="unqualified_model_name")

def test_should_download(deploy: EzDeploy) -> None:
    assert deploy is not None
    
def test_should_deploy(deploy: EzDeploy) -> None:
    assert deploy is not None
    
def test_should_register(deploy: EzDeploy) -> None:
    assert deploy is not None