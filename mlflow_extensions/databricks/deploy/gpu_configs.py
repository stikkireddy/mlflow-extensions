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


class AwsVMConfigs(Enum):
    G5_16XLARGE = GPUConfig(
        name="g5.16xlarge", gpu_count=1, gpu_type=GPUType.A10G, cloud=Cloud.AWS
    )
    G5_24XLARGE = GPUConfig(
        name="g5.24xlarge", gpu_count=4, gpu_type=GPUType.A10G, cloud=Cloud.AWS
    )
    G5_48XLARGE = GPUConfig(
        name="g5.48xlarge", gpu_count=8, gpu_type=GPUType.A10G, cloud=Cloud.AWS
    )
    P4D_24XLARGE = GPUConfig(
        name="p4d.24xlarge", gpu_count=8, gpu_type=GPUType.A100_80G, cloud=Cloud.AWS
    )
    P4DE_24XLARGE = GPUConfig(
        name="p4de.24xlarge", gpu_count=8, gpu_type=GPUType.A100_80G, cloud=Cloud.AWS
    )


class AzureVMConfigs(Enum):
    NV36ADS_A10_V5 = GPUConfig(
        name="Standard_NV36ads_A10_v5",
        gpu_count=1,
        gpu_type=GPUType.A10G,
        cloud=Cloud.AZURE,
    )
    NV72ADS_A10_V5 = GPUConfig(
        name="Standard_NV72ads_A10_v5",
        gpu_count=2,
        gpu_type=GPUType.A10G,
        cloud=Cloud.AZURE,
    )
    NC24ADS_A100_V4 = GPUConfig(
        name="Standard_NC24ads_A100_v4",
        gpu_count=1,
        gpu_type=GPUType.A100_80G,
        cloud=Cloud.AZURE,
    )
    NC48ADS_A100_V4 = GPUConfig(
        name="Standard_NC48ads_A100_v4",
        gpu_count=2,
        gpu_type=GPUType.A100_80G,
        cloud=Cloud.AZURE,
    )
    NC96ADS_A100_V4 = GPUConfig(
        name="Standard_NC96ads_A100_v4",
        gpu_count=4,
        gpu_type=GPUType.A100_80G,
        cloud=Cloud.AZURE,
    )


class GCPVMConfigs(Enum):
    A2_HIGHGPU_1G = GPUConfig(
        name="a2-highgpu-1g", gpu_count=1, gpu_type=GPUType.A100_40G, cloud=Cloud.GCP
    )
    A2_HIGHGPU_2G = GPUConfig(
        name="a2-highgpu-2g", gpu_count=2, gpu_type=GPUType.A100_40G, cloud=Cloud.GCP
    )
    A2_HIGHGPU_4G = GPUConfig(
        name="a2-highgpu-4g", gpu_count=4, gpu_type=GPUType.A100_40G, cloud=Cloud.GCP
    )
    A2_HIGHGPU_8G = GPUConfig(
        name="a2-highgpu-8g", gpu_count=8, gpu_type=GPUType.A100_40G, cloud=Cloud.GCP
    )
    A2_MEGAGPU_16G = GPUConfig(
        name="a2-megagpu-16g", gpu_count=16, gpu_type=GPUType.A100_40G, cloud=Cloud.GCP
    )


ALL_VALID_VM_CONFIGS = (
    [gpu.value for gpu in GCPVMConfigs]
    + [gpu.value for gpu in AzureVMConfigs]
    + [gpu.value for gpu in AwsVMConfigs]
)
