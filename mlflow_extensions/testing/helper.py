import functools
import inspect
import os
import subprocess
from enum import Enum

import psutil

from mlflow_extensions.log import Logger, get_logger

LOGGER: Logger = get_logger()


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

        client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="local")
        return func(*args, client=client, **kwargs)

    return wrapper


class ServerFramework(Enum):
    VLLM = "vllm"
    SGLANG = "sglang"


def get_process_ids(search_string):
    try:
        # Run 'ps aux' to get the process list
        result = subprocess.run(
            ["ps", "aux"], text=True, capture_output=True, check=True
        )

        # Process the output
        pids = []
        for line in result.stdout.splitlines():
            if search_string in line:
                # Extract PID (assumed to be in the second column)
                LOGGER.info(f"Found orphaned process: {line} matching {search_string}")
                parts = line.split()
                if len(parts) > 1 and parts[1].isdigit():
                    pids.append(parts[1])

        return pids

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return []


def kill_process(pid):
    try:
        subprocess.run(["kill", "-9", pid], check=True)
        print(f"Killed process with PID {pid}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to kill process with PID {pid}: {e}")


def kill_processes_containing(search_string):
    pids = get_process_ids(search_string)
    for pid in pids:
        kill_process(pid)


def is_process_active(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        status = process.status()
        if status == psutil.STATUS_ZOMBIE:
            print(f"Process {pid} is a zombie.")
            return False
        else:
            print(f"Process {pid} is running with status: {status}.")
            return True
    except psutil.NoSuchProcess:
        print(f"Process {pid} does not exist (killed or never existed).")
        return False
    except psutil.AccessDenied:
        print(f"Access denied to process {pid}.")
        return False
