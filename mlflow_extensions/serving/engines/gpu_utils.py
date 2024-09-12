import shutil


def get_gpu_count(default: int = 1) -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        # we are assuming that there is atleast one gpu for when you need to use the engines
        return default


def get_total_shm_size_gb():
    shm_path = "/dev/shm"
    total, used, free = shutil.disk_usage(shm_path)
    return total / (1024**3)


def not_enough_shm(n_gb: int = 1) -> bool:
    try:
        return get_total_shm_size_gb() < n_gb
    except FileNotFoundError:
        return True
