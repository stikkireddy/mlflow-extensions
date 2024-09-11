import fnmatch
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployConfig
from mlflow_extensions.databricks.deploy.gpu_configs import GPUConfig
from mlflow_extensions.databricks.prebuilt import prebuilt
from mlflow_extensions.serving.engines import VLLMEngineConfig
from mlflow_extensions.serving.engines.base import EngineProcess
from mlflow_extensions.testing.audio_basic import (
    encode_audio_base64_from_url,
    query_audio,
)
from mlflow_extensions.testing.helper import ServerFramework, kill_processes_containing
from mlflow_extensions.testing.text_basic import query_text
from mlflow_extensions.testing.vision_basic import (
    query_vision,
    query_vision_multi_input,
)


def make_process_and_get_artifacts(
    config: EzDeployConfig, local_dir=None
) -> Tuple[EngineProcess, dict]:
    if local_dir is not None:
        artifacts = config.engine_config.setup_artifacts(local_dir)
    else:
        artifacts = config.engine_config.setup_artifacts()
    engine = config.engine_proc(config=config.engine_config)
    return engine, artifacts


@dataclass
class RequestResult:
    model: str
    config_name: str
    framework: str
    gpu: str
    output: str
    error_msg: str
    is_error: bool
    did_server_crash: bool
    cloud: str
    server_command: str

    @staticmethod
    def make_df_friendly(results: List["RequestResult"]):
        return [asdict(result) for result in results]


class ModelContextRunner:
    def __init__(
        self,
        ez_config: EzDeployConfig,
        current_gpu: GPUConfig,
        skip_health_check: bool = True,
    ):
        self.ez_config = ez_config
        self.engine: Optional[EngineProcess] = None
        self.artifacts: Optional[dict] = None
        self.model_context = None
        self._results: List[RequestResult] = []
        self.current_gpu = current_gpu
        self.command: Optional[str] = None
        self._skip_health_check = skip_health_check

    def __enter__(self):
        try:
            from mlflow.pyfunc import PythonModelContext
        except ImportError:
            raise ImportError(
                "mlflow is required to use this class run pip install -U mlflow"
            )
        self.engine, self.artifacts = make_process_and_get_artifacts(self.ez_config)
        self.add_success(result="SUCCESSFULLY LOADED ARTIFACTS")
        self.model_context = PythonModelContext(
            artifacts=self.artifacts, model_config={}
        )
        self.ez_config.engine_config.to_run_command(self.model_context)
        self.engine.start_proc(
            self.model_context, health_check_thread=not self._skip_health_check
        )
        self.add_success(result="SUCCESSFULLY STARTED SERVER")
        return self

    def add_error(self, *, error_msg: str):
        self.results.append(
            RequestResult(
                model=self.ez_config.engine_config.model,
                config_name=self.ez_config.name,
                gpu=f"{self.current_gpu.name}x{self.current_gpu.gpu_count}",
                output="",
                error_msg=error_msg,
                is_error=True,
                cloud=self.current_gpu.cloud.value,
                did_server_crash=self.engine.is_process_healthy() is False,
                framework=self.ez_config.engine_proc.__name__,
                server_command=self.command,
            )
        )

    def add_success(self, *, result: str):
        self.results.append(
            RequestResult(
                model=self.ez_config.engine_config.model,
                config_name=self.ez_config.name,
                gpu=f"{self.current_gpu.name}x{self.current_gpu.gpu_count}",
                error_msg="",
                output=result,
                is_error=False,
                cloud=self.current_gpu.cloud.value,
                did_server_crash=self.engine.is_process_healthy() is False,
                framework=self.ez_config.engine_proc.__name__,
                server_command=self.command,
            )
        )

    @property
    def results(self):
        return self._results

    @property
    def results_as_dict(self):
        return [asdict(result) for result in self._results]

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.add_error(error_msg=str(exc_val))

        if self.engine is not None and self.engine.server_process is not None:
            self.engine.stop_proc()


def run_all_tests(
    *,
    gpu_config: GPUConfig,
    server_framework: ServerFramework,
    model_filter_fnmatch_str: str = "*",
    ezconfig_name_filter_fnmatch_str: str = "*",
) -> List[RequestResult]:
    # gettysburg.wav is a 17 second audio file
    audio_data = encode_audio_base64_from_url(
        "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg.wav"
    )
    results = []

    for modality, model_arg in prebuilt.__dict__.items():
        for serving_type, serving_framework_cfg in model_arg.__dict__.items():
            for key, ezconfig in serving_framework_cfg.__dict__.items():
                ezconfig: EzDeployConfig
                if serving_type != server_framework.value:
                    continue
                if (
                    ezconfig.serving_config.minimum_memory_in_gb
                    > gpu_config.total_memory_gb
                ):
                    continue
                if (
                    fnmatch.fnmatch(
                        ezconfig.engine_config.model, model_filter_fnmatch_str
                    )
                    is False
                ):
                    print(
                        f"SKIPPING {ezconfig.engine_config.model} because it does not match "
                        f"{model_filter_fnmatch_str}"
                    )
                    continue
                if (
                    fnmatch.fnmatch(ezconfig.name, ezconfig_name_filter_fnmatch_str)
                    is False
                ):
                    print(
                        f"SKIPPING {ezconfig.engine_config.model} because it does not match "
                        f"{ezconfig_name_filter_fnmatch_str}"
                    )
                    continue

                # kill everything before starting a new process
                # we want to make sure ports are available
                kill_processes_containing("sglang.launch_server")
                kill_processes_containing("vllm.entrypoints.openai.api_server")
                with ModelContextRunner(
                    ez_config=ezconfig, current_gpu=gpu_config
                ) as ctx:
                    query_audio(
                        ctx=ctx,
                        model=ezconfig.engine_config.model,
                        modality_type=modality,
                        audio_data=audio_data,
                    )
                    query_vision(
                        ctx=ctx,
                        model=ezconfig.engine_config.model,
                        modality_type=modality,
                    )
                    query_text(
                        ctx=ctx,
                        model=ezconfig.engine_config.model,
                        modality_type=modality,
                    )
                    if (
                        server_framework == ServerFramework.VLLM
                        and isinstance(ezconfig.engine_config, VLLMEngineConfig)
                        and ezconfig.engine_config.max_num_images is not None
                        and ezconfig.engine_config.max_num_images > 1
                    ):
                        # TODO: accept max num images to ensure we test multiple images
                        query_vision_multi_input(
                            ctx=ctx,
                            model=ezconfig.engine_config.model,
                            modality_type=modality,
                        )

                results.extend(ctx.results)
                print("WAITING FOR SERVER TO CLEANLY CLOSE BEFORE STARTING NEW SERVER")
                time.sleep(5)

    return results
