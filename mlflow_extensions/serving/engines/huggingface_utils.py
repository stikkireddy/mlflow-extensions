import json
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as e:
    print(
        "Error importing huggingface_hub module please run pip install huggingface_hub or upgrade the sdk "
        "by running pip install huggingface_hub --upgrade"
    )
    raise e


def is_repo_local_dir(repo_id: str):
    return Path(repo_id).is_dir()


def snapshot_download_local(repo_id: str, local_dir: str, tokenizer_only: bool = False):
    if is_repo_local_dir(repo_id):
        # TODO: add a check to see if the path is a volume path and then copy the files to /local_disk0
        return repo_id
    local_dir = local_dir.rstrip("/")
    model_local_path = f"{local_dir}/{repo_id}"
    kwargs = {}
    if tokenizer_only:
        kwargs["ignore_patterns"] = ["*.bin", "*.safetensors"]
    snapshot_download(repo_id=repo_id, local_dir=model_local_path, **kwargs)
    return model_local_path


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
