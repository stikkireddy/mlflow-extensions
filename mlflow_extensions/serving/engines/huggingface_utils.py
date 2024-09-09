import json
import shutil
import os
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, scan_cache_dir
except ImportError as e:
    print(
        "Error importing huggingface_hub module please run pip install huggingface_hub or upgrade the sdk "
        "by running pip install huggingface_hub --upgrade"
    )
    raise e


def snapshot_download_local(repo_id: str, local_dir: str, tokenizer_only: bool = False):
    local_dir = local_dir.rstrip("/")
    model_local_path = f"{local_dir}/{repo_id}"
    kwargs = {}
    if tokenizer_only:
        kwargs["ignore_patterns"] = ["*.bin", "*.safetensors"]
    snapshot_download(repo_id=repo_id, local_dir=model_local_path, **kwargs)
    return model_local_path


def get_local_snapshot(repo_id, cache_dir, local_dir):
    """
    Scans local_dir for a model matching the repo_id and returns the path or None.
    Also supports non-cache directories.
    """
    snapshot_path = None

    try:
        scan = scan_cache_dir(cache_dir=cache_dir)
        for repo in [r for r in scan.repos if r.repo_id == repo_id]:
            for rev in repo.revisions:
                snapshot_path = rev.snapshot_path
                break
            if snapshot_path:
                break

    except Exception as e:
        print(f"Cache directory scan failed: {e}")

        # If scan_cache_dir doesn't work, check if the directory contains config.json
        if os.path.exists(local_dir):
            config_path = os.path.join(local_dir, "config.json")
            if os.path.exists(config_path):
                snapshot_path = local_dir

        if not snapshot_path:
            print("No valid model directory found.")

    finally:
        # If a snapshot was found and local_dir is specified, copy the contents
        if snapshot_path:
            try:
                shutil.copytree(snapshot_path, local_dir, dirs_exist_ok=True)
                print(f"Model copied to {local_dir}")
                return local_dir
            except Exception as e:
                print(f"Failed to copy model to {local_dir}: {e}")

    return snapshot_path


def ensure_chat_template(tokenizer_file: str, chat_template_key: str = "chat_template"):
    tokenizer_file_path = Path(tokenizer_file)
    assert (
        tokenizer_file_path.exists()
    ), f"Tokenizer config file not found at {tokenizer_file}"
    with open(tokenizer_file, "r") as f:
        tokenizer_config = json.loads(f.read())
        chat_template = tokenizer_config.get(chat_template_key)
        if chat_template is None:
            raise ValueError(
                f"Chat template not found in tokenizer config file {tokenizer_file}"
            )
