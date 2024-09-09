import pandas as pd

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployConfig
from mlflow_extensions.databricks.deploy.gpu_configs import Cloud
from mlflow_extensions.databricks.prebuilt import prebuilt
from mlflow_extensions.serving.engines import VLLMEngineConfig

try:
    import tabulate
except ImportError as e:
    print("Please install tabulate to run this script. pip install tabulate")
    raise e

if __name__ == "__main__":
    data = []
    for model_type, model_arg in prebuilt.__dict__.items():
        for serving_type, serving_framework_cfg in model_arg.__dict__.items():
            for key, value in serving_framework_cfg.__dict__.items():
                value: EzDeployConfig
                context_length = "Default"
                if isinstance(value.engine_config, VLLMEngineConfig):
                    if value.engine_config.max_model_len is not None:
                        context_length = str(value.engine_config.max_model_len)
                azure_gpu_cfg = value.serving_config.smallest_gpu(Cloud.AZURE)
                aws_gpu_cfg = value.serving_config.smallest_gpu(Cloud.AWS)
                data.append(
                    {
                        "model_type": model_type,
                        "cfg_path": f"prebuilt.{model_type}.{serving_type}.{key}",
                        "huggingface_link": f"https://huggingface.co/{value.engine_config.model}",
                        "context_length": context_length,
                        "min_azure_ep_type_gpu": f"{azure_gpu_cfg.name} [{azure_gpu_cfg.gpu_type.value}x{azure_gpu_cfg.gpu_count} {azure_gpu_cfg.total_memory_gb}GB]",
                        "min_aws_ep_type_gpu": f"{aws_gpu_cfg.name} [{aws_gpu_cfg.gpu_type.value}x{aws_gpu_cfg.gpu_count} {aws_gpu_cfg.total_memory_gb}GB]",
                    }
                )

    df = pd.DataFrame(data)
    print(df.to_markdown(index=False))
