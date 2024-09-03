from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployConfig


def make_process_and_get_artifacts(config: EzDeployConfig, local_dir=None):
    if local_dir is not None:
        artifacts = config.engine_config.setup_artifacts(local_dir)
    else:
        artifacts = config.engine_config.setup_artifacts()

    engine = config.engine_proc(config=config.engine_config)

    return engine, artifacts
