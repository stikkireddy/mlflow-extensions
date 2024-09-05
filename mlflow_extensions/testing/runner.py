import functools
import inspect
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Tuple, Optional, List, Type

from mlflow.pyfunc import PythonModelContext

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployConfig
from mlflow_extensions.databricks.deploy.gpu_configs import (
    GPUConfig,
    AzureServingGPUConfig,
)
from mlflow_extensions.databricks.prebuilt import prebuilt
from mlflow_extensions.serving.engines.base import EngineProcess
from mlflow_extensions.testing.audio_basic import (
    query_audio,
    encode_audio_base64_from_url,
)
from mlflow_extensions.testing.text_basic import query_text
from mlflow_extensions.testing.vision_basic import query_vision


class Modality(Enum):
    VISION = "vision"
    TEXT = "text"
    AUDIO = "audio"


class ServerFramework(Enum):
    VLLM = "vllm"
    SGLANG = "sglang"


def run_if(modality: Modality):
    def outer(func):

        expected_kwargs = ["modality"]
        for kwarg in expected_kwargs:
            if kwarg not in inspect.getfullargspec(func).kwonlyargs:
                raise ValueError(
                    f"Function must have {expected_kwargs} as keyword-only arguments"
                )

        @functools.wraps(func)
        def inner(*args, **kwargs):
            this_modality = kwargs.get("modality")
            if modality == this_modality:
                return func(*args, **kwargs)

            print(
                f"Skipping {func.__name__} because modality is {this_modality} not {modality}"
            )
            return

        return inner

    return outer


def inject_openai_client(func):
    # error if function has regular args
    has_args = inspect.getfullargspec(func).args
    if has_args:
        raise ValueError("Function cannot have regular arguments")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        host = kwargs.get("host", "0.0.0.0")
        port = kwargs.get("port", 9989)
        from openai import OpenAI

        client = OpenAI(base_url=f"http://{host}:{port}", api_key="local")
        return func(client=client, **kwargs)

    return wrapper


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
    framework: str
    gpu: str
    output: str
    error_msg: str
    is_error: bool
    did_server_crash: bool
    cloud: str
    server_command: str


class ModelContextRunner:
    def __init__(self, ez_config: EzDeployConfig, current_gpu: GPUConfig):
        self.ez_config = ez_config
        self.engine: Optional[EngineProcess] = None
        self.artifacts: Optional[dict] = None
        self.model_context = None
        self._results: List[RequestResult] = []
        self.current_gpu = current_gpu
        self.command: Optional[str] = None

    def __enter__(self):
        try:
            from mlflow.pyfunc import PythonModelContext
        except ImportError:
            raise ImportError(
                "mlflow is required to use this class run pip install -U mlflow"
            )
        self.engine, self.artifacts = make_process_and_get_artifacts(self.ez_config)
        self.model_context = PythonModelContext(
            artifacts=self.artifacts, model_config={}
        )
        self.ez_config.engine_config.to_run_command(self.model_context)
        self.engine.start_proc(self.model_context)

    def add_error(self, *, error_msg: str):
        self.results.append(
            RequestResult(
                model=self.ez_config.engine_config.model,
                gpu=f"{self.current_gpu.name}x{self.current_gpu.gpu_count}",
                output="",
                error_msg=error_msg,
                is_error=True,
                cloud=self.current_gpu.cloud.value,
                did_server_crash=self.engine.is_process_healthy() is False,
                framework=self.ez_config.engine_proc.__class__.__name__,
                server_command=self.command,
            )
        )

    def add_success(self, *, result: str):
        self.results.append(
            RequestResult(
                model=self.ez_config.engine_config.model,
                gpu=f"{self.current_gpu.name}x{self.current_gpu.gpu_count}",
                error_msg="",
                output=result,
                is_error=False,
                cloud=self.current_gpu.cloud.value,
                did_server_crash=self.engine.is_process_healthy() is False,
                framework=self.ez_config.engine_proc.__class__.__name__,
                server_command=self.command,
            )
        )

    @property
    def results(self):
        return self._results

    @property
    def results_as_dict(self):
        return [asdict(result) for result in self._results]

    def _cleanup(self):
        if self.engine is not None:
            self.engine.start_proc(self.model_context)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.engine is not None and self.engine.server_process is not None:
            self.engine.stop_proc()


def run_all_tests(*, gpu_config: GPUConfig, server_framework: ServerFramework) -> List[RequestResult]:
    # gettysburg.wav is a 17 second audio file
    audio_data = encode_audio_base64_from_url(
        "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg.wav"
    )
    results = []

    for modality, model_arg in prebuilt.__dict__.items():
        for serving_type, serving_framework_cfg in model_arg.__dict__.items():
            for key, ezconfig in serving_framework_cfg.__dict__.items():
                if serving_type == server_framework.value:
                    if (
                        ezconfig.serving_config.minimum_memory_in_gb
                        <= gpu_config.gpu_type.memory_gb
                    ):
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

                        results.extend(ctx.results)

    return results