import functools
import inspect
from enum import Enum


class Modality(Enum):
    VISION = "vision"
    TEXT = "text"
    AUDIO = "audio"


def run_if(modality: Modality):
    def outer(func):

        @functools.wraps(func)
        def inner(*args, **kwargs):
            this_modality = kwargs.get("modality_type")
            if this_modality is None:
                raise ValueError("modality_type must be provided")
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


class ServerFramework(Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
