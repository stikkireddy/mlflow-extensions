from dataclasses import dataclass
from enum import Enum


class Cloud(Enum):
    AWS = "AWS"
    AZURE = "AZURE"
    GCP = "GCP"

    @staticmethod
    def from_host(host):
        if "gcp.databricks.com" in host:
            return Cloud.GCP
        if "azuredatabricks.net" in host:
            return Cloud.AZURE
        return Cloud.AWS


class GPUType(Enum):
    T4 = "T4"
    A10G = "A10G"
    A100_80G = "A100_80G"
    A100_40G = "A100_40G"
    H100 = "H100"

    @property
    def memory_gb(self):
        memory_map = {
            "T4": 16,
            "A10G": 24,
            "A100_80G": 80,
            "A100_40G": 40,
            "H100": 80,
        }
        return memory_map.get(self.value, "Unknown")


@dataclass(frozen=True, kw_only=True)
class GPUConfig:
    name: str
    gpu_count: int
    gpu_type: GPUType
    cloud: Cloud

    def __repr__(self):
        return f"{self.name}({self.gpu_count}x{self.gpu_type})"

    @property
    def total_memory_gb(self):
        return self.gpu_count * self.gpu_type.memory_gb

    @property
    def single_gpu_memory_gb(self):
        return self.gpu_type.memory_gb


class AWSServingGPUConfig(Enum):
    GPU_SMALL = GPUConfig(
        name="GPU_SMALL", gpu_count=1, gpu_type=GPUType.T4, cloud=Cloud.AWS
    )  # 1xT4
    GPU_MEDIUM = GPUConfig(
        name="GPU_MEDIUM", gpu_count=1, gpu_type=GPUType.A10G, cloud=Cloud.AWS
    )  # 1xA10G
    MULTIGPU_MEDIUM = GPUConfig(
        name="MULTIGPU_MEDIUM", gpu_count=4, gpu_type=GPUType.A10G, cloud=Cloud.AWS
    )  # 4xA10G
    GPU_MEDIUM_8 = GPUConfig(
        name="GPU_MEDIUM_8", gpu_count=8, gpu_type=GPUType.A10G, cloud=Cloud.AWS
    )  # 8xA10G
    GPU_LARGE_8 = GPUConfig(
        name="GPU_LARGE_8", gpu_count=8, gpu_type=GPUType.A100_80G, cloud=Cloud.AWS
    )  # 8xA100_80G
    GPU_XLARGE_8 = GPUConfig(
        name="GPU_XLARGE_8", gpu_count=8, gpu_type=GPUType.H100, cloud=Cloud.AWS
    )  # 8xH100


class AzureServingGPUConfig(Enum):
    GPU_SMALL = GPUConfig(
        name="GPU_SMALL", gpu_count=1, gpu_type=GPUType.T4, cloud=Cloud.AZURE
    )  # 1xT4
    GPU_LARGE = GPUConfig(
        name="GPU_LARGE", gpu_count=1, gpu_type=GPUType.A100_80G, cloud=Cloud.AZURE
    )  # 1xA100_80G
    GPU_LARGE_2 = GPUConfig(
        name="GPU_LARGE_2", gpu_count=2, gpu_type=GPUType.A100_80G, cloud=Cloud.AZURE
    )  # 2xA100_80G
    GPU_LARGE_8 = GPUConfig(
        name="GPU_LARGE_8", gpu_count=8, gpu_type=GPUType.A100_80G, cloud=Cloud.AZURE
    )  # 8xA100_80G


ALL_VALID_GPUS = [gpu.value for gpu in AzureServingGPUConfig] + [
    gpu.value for gpu in AWSServingGPUConfig
]
