import json
from dataclasses import dataclass, asdict, field
from typing import Type, Optional, Literal, List

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

from mlflow_extensions.databricks.deploy.gpu_configs import (
    GPUConfig,
    Cloud,
    ALL_VALID_GPUS,
)
from mlflow_extensions.serving.engines.base import EngineConfig, EngineProcess
from mlflow_extensions.serving.wrapper import (
    CustomServingEnginePyfuncWrapper,
    DIAGNOSTICS_REQUEST_KEY,
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

    def download(self):
        self._model = CustomServingEnginePyfuncWrapper(
            engine=self._config.engine_proc, engine_config=self._config.engine_config
        )
        self._model.setup()
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

    def deploy(
        self,
        name,
        *,
        scale_to_zero: bool = True,
        workload_size: Literal["Small", "Medium", "Large"] = "Small",
        workload_type: Optional[str] = None,
        enable_diagnostics: bool = False,
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
        environment_vars = {}
        if enable_diagnostics is True:
            environment_vars[DIAGNOSTICS_REQUEST_KEY] = "true"
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
