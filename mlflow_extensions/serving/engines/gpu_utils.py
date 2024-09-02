def get_gpu_count(default: int = 1) -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        # we are assuming that there is atleast one gpu for when you need to use the engines
        return default
