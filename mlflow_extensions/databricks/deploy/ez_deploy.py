import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Type

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, ResourceDoesNotExist
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

from mlflow_extensions.databricks.deploy.ez_deploy_lite import EzDeployLiteManager
from mlflow_extensions.databricks.deploy.gpu_configs import (
    ALL_VALID_GPUS,
    Cloud,
    GPUConfig,
)
from mlflow_extensions.log import LogConfig
from mlflow_extensions.serving.engines.base import EngineConfig, EngineProcess
from mlflow_extensions.serving.wrapper import (
    ARCHIVE_LOG_PATH_KEY,
    ENABLE_DIAGNOSTICS_FLAG,
    LOG_FILE_KEY,
    CustomServingEnginePyfuncWrapper,
)


@dataclass(kw_only=True, frozen=True)
class ServingConfig:
    valid_gpus: List[GPUConfig] = field(default_factory=lambda: ALL_VALID_GPUS)
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


@dataclass(frozen=True, kw_only=True)
class EzDeployConfig:
    name: str
    engine_config: EngineConfig
    engine_proc: Type[EngineProcess]
    serving_config: ServingConfig
    pip_config_override: List[str] = None

    def serialize_json(self):
        data = {
            "name": self.name,
            "engine_config": asdict(self.engine_config),
            "engine_proc": self.engine_proc.__name__,
            "pip_config_override": self.pip_config_override,
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        engine_proc = data["engine_proc"]
        if engine_proc == "VLLMEngineProcess":
            from mlflow_extensions.serving.engines import (
                VLLMEngineConfig,
                VLLMEngineProcess,
            )

            engine_proc = VLLMEngineProcess
            engine_config = VLLMEngineConfig(**data["engine_config"])
        elif engine_proc == "SglangEngineProcess":
            from mlflow_extensions.serving.engines import (
                SglangEngineConfig,
                SglangEngineProcess,
            )

            engine_proc = SglangEngineProcess
            engine_config = SglangEngineConfig(**data["engine_config"])
        else:
            raise ValueError(f"Unsupported engine process {engine_proc}")
        return EzDeployConfig(
            name=data["name"],
            engine_config=engine_config,
            engine_proc=engine_proc,
            pip_config_override=data["pip_config_override"],
            serving_config=ServingConfig(minimum_memory_in_gb=-1),
        )

    def to_proc(self) -> EngineProcess:
        return self.engine_proc(config=self.engine_config)

    def download_artifacts(self, local_dir: str = "/local_disk0/models"):
        return self.engine_config.setup_artifacts(local_dir=local_dir)


class EzDeploy:

    def __init__(
        self,
        *,
        config: EzDeployConfig,
        registered_model_name: str,
        databricks_host: str = None,
        databricks_token: str = None,
    ):
        self._config = config
        self._registered_model_name = registered_model_name
        assert len(self._registered_model_name.split(".")) == 3, (
            "Ensure that your registered model name "
            "follows the level namespace;"
            " <catalog>.<schema>.<model_name>"
        )
        self._downloaded = False
        self._model: Optional[CustomServingEnginePyfuncWrapper] = None
        self._latest_registered_model_version = None
        if databricks_host is None or databricks_token is None:
            from mlflow.utils.databricks_utils import get_databricks_host_creds

            self._client = WorkspaceClient(
                host=get_databricks_host_creds().host,
                token=get_databricks_host_creds().token,
            )
            self._cloud = Cloud.from_host(get_databricks_host_creds().host)
        else:
            self._client = WorkspaceClient(host=databricks_host, token=databricks_token)
            self._cloud = Cloud.from_host(databricks_host)

    def download(self, *, local_dir=None):
        self._model = CustomServingEnginePyfuncWrapper(
            engine=self._config.engine_proc, engine_config=self._config.engine_config
        )
        self._model.setup(local_dir=local_dir)
        self._downloaded = True

    def register(self):
        assert self._downloaded is True, "Ensure you run the download method first"
        assert self._model is not None, "Ensure you run download"
        assert (
            self._registered_model_name is not None
        ), "Ensure you provide a valid registered_name"

        import mlflow

        mlflow.set_registry_uri("databricks-uc")

        from mlflow.models import infer_signature

        with mlflow.start_run():
            logged_model = mlflow.pyfunc.log_model(
                "model",
                python_model=self._model,
                artifacts=self._model.artifacts,
                pip_requirements=self._config.pip_config_override
                or self._model.get_pip_reqs(),
                signature=infer_signature(
                    model_input=["string-request"], model_output=["string-response"]
                ),
                registered_model_name=self._registered_model_name,
            )
            mlflow.log_params(
                {
                    "model": self._config.engine_config.model,
                    "host": self._config.engine_config.host,
                    "port": self._config.engine_config.port,
                    "command": str(self._config.engine_config.to_run_command()),
                    "serialized_engine_config": str(
                        json.dumps(asdict(self._config.engine_config))
                    ),
                }
            )
            self._latest_registered_model_version = (
                logged_model.registered_model_version
            )

    def _does_endpoint_exist(self, name) -> bool:
        try:
            self._client.serving_endpoints.get(name)
            return True
        except ResourceDoesNotExist:
            return False

    def _throw_if_volume_does_not_exist(self, path: str) -> None:
        def _path_to_volume(path: str) -> str:
            pattern: str = r"^(?:dbfs:)?/Volumes/([^/]+)/([^/]+)/([^/]+)/?.*"
            match: re.Match = re.match(pattern, path)
            if match:
                catalog: str = match.group(1)
                schema: str = match.group(2)
                volume: str = match.group(3)
                return f"{catalog}.{schema}.{volume}"
            raise ValueError(f"Invalid path {path}")

        try:
            volume: str = _path_to_volume(path)
            self._client.volumes.read(volume)
        except NotFound:
            raise ValueError(f"Volume {volume} does not exist")

    def deploy(
        self,
        name,
        *,
        scale_to_zero: bool = True,
        workload_size: Literal["Small", "Medium", "Large"] = "Small",
        workload_type: Optional[str] = None,
        enable_diagnostics: bool = False,
        log_config: Optional[LogConfig] = None,
    ):
        gpu_cfg = self._config.serving_config.smallest_gpu(self._cloud)
        endpoint_exists = self._does_endpoint_exist(name)
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession()
            workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
            print(
                f"Deploying model: {name} track progress here: https://{workspace_url}/ml/endpoints/{name}"
            )
        except ImportError:
            print(
                f"Deploying model: {name}; look at the serving tab to track progress..."
            )
        environment_vars = {
            # ensure that for docker container that all logs are always flushed to terminal
            # and not buffered anywhere
            "PYTHONUNBUFFERED": "1",
        }
        if enable_diagnostics is True:
            environment_vars[ENABLE_DIAGNOSTICS_FLAG] = "true"
        if log_config is not None:
            log_file: str = log_config.filename or f"{name}.log"
            environment_vars[LOG_FILE_KEY] = log_file
            if log_config.archive_path is not None:
                self._throw_if_volume_does_not_exist(log_config.archive_path)
                archive_log_path: Path = Path(log_config.archive_path)
                archive_log_path = (
                    archive_log_path
                    if archive_log_path.name == name
                    else archive_log_path / name
                )
                environment_vars[ARCHIVE_LOG_PATH_KEY] = archive_log_path.as_posix()
        if endpoint_exists is False:
            self._client.serving_endpoints.create(
                name=name,
                config=EndpointCoreConfigInput(
                    name=name,
                    served_entities=[
                        ServedEntityInput(
                            name=name,
                            entity_name=self._registered_model_name,
                            entity_version=self._latest_registered_model_version,
                            scale_to_zero_enabled=scale_to_zero,
                            workload_type=workload_type or gpu_cfg.name,
                            workload_size=workload_size,
                            environment_vars=environment_vars,
                        )
                    ],
                ),
            )
        else:
            self._client.serving_endpoints.update_config(
                name=name,
                served_entities=[
                    ServedEntityInput(
                        name=name,
                        entity_name=self._registered_model_name,
                        entity_version=self._latest_registered_model_version,
                        scale_to_zero_enabled=scale_to_zero,
                        workload_type=workload_type or gpu_cfg.name,
                        workload_size=workload_size,
                        environment_vars=environment_vars,
                    )
                ],
            )


class EzDeployLite:

    def __init__(
        self,
        ez_deploy_config: EzDeployConfig,
        databricks_host: str = None,
        databricks_token: str = None,
    ):
        self._config: EzDeployConfig = ez_deploy_config
        self._downloaded = False
        if databricks_host is None or databricks_token is None:
            from mlflow.utils.databricks_utils import get_databricks_host_creds

            self._client = WorkspaceClient(
                host=get_databricks_host_creds().host,
                token=get_databricks_host_creds().token,
            )
            self._cloud = Cloud.from_host(get_databricks_host_creds().host)
        else:
            self._client = WorkspaceClient(host=databricks_host, token=databricks_token)
            self._cloud = Cloud.from_host(databricks_host)
        self._edlm = EzDeployLiteManager(
            databricks_host=databricks_host, databricks_token=databricks_token
        )

    def deploy(
        self,
        deployment_name: str,
        hf_secret_scope: str,
        hf_secret_key: str,
        specific_git_ref: str = None,
    ):
        self._edlm.upsert(
            deployment_name,
            cloud_provider=Cloud.GCP,
            ez_deploy_config=self._config,
            hf_secret_key=hf_secret_key,
            hf_secret_scope=hf_secret_scope,
            entrypoint_git_ref=specific_git_ref,
        )
        self._edlm.start_server(deployment_name)
