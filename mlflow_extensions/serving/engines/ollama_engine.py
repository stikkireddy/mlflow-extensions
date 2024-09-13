import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from mlflow.pyfunc import PythonModelContext

from mlflow_extensions.log import Logger, get_logger
from mlflow_extensions.serving.engines.base import Command, EngineConfig, EngineProcess

LOGGER: Logger = get_logger()


def set_full_permissions(path: str):
    """Recursively set full permissions for all files and directories in the given path."""
    for root, dirs, files in os.walk(path):
        for d in dirs:
            dir_path = os.path.join(root, d)
            os.chmod(dir_path, 0o777)  # Full permissions for directories
        for f in files:
            file_path = os.path.join(root, f)
            os.chmod(file_path, 0o777)  # Full permissions for files


def download_and_extract(
    version: str = "0.3.8", download_dir: str = ".", extract_dir: str = "ollama"
) -> Optional[str]:
    version = version.lstrip("v")
    url = f"https://github.com/ollama/ollama/releases/download/v{version}/ollama-linux-amd64.tgz"
    downloaded_file = os.path.join(download_dir, "ollama-linux-amd64.tgz")
    extract_path = os.path.join(download_dir, extract_dir)

    os.makedirs(download_dir, exist_ok=True)

    try:
        subprocess.run(["wget", url, "-O", downloaded_file], check=True)
        print(f"Downloaded {downloaded_file} from {url}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")
        return

    try:
        subprocess.run(
            ["tar", "-xzvf", downloaded_file, "-C", download_dir], check=True
        )
        print(f"Extracted {downloaded_file} into {download_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting file: {e}")
        return

    if os.path.exists(downloaded_file):
        os.remove(downloaded_file)
        print(f"Removed the downloaded file {downloaded_file}")

    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
        print(f"Removed the previous extracted directory {extract_path}")

    extracted_folder_name = "ollama-linux-amd64"
    extracted_folder_path = os.path.join(download_dir, extracted_folder_name)

    if os.path.exists(extracted_folder_path):
        os.rename(extracted_folder_path, extract_path)
        print(f"Renamed the extracted directory to {extract_path}")
    else:
        print(
            f"Expected directory {extracted_folder_path} not found. No renaming performed."
        )

    set_full_permissions(str(download_dir))

    return str((Path(download_dir) / "bin/ollama").absolute())


@dataclass(frozen=True, kw_only=True)
class OllamaEngineConfig(EngineConfig):
    ollama_version: str = field(default="0.3.8")
    model_download_dir_name: str = field(default="ollama-downloaded-models")
    model_artifact_key: str = field(default="model")
    root_folder_name: str = field(default="ollama")

    def setup_artifacts(self, local_dir: str = "/root/models") -> Dict[str, str]:
        # Example usage:
        download_dir = Path(local_dir) / self.root_folder_name
        ollama_cli = download_and_extract(self.ollama_version, str(download_dir))
        new_env = os.environ.copy()
        new_env.update(
            {
                "OLLAMA_HOST": f"{self.host}:{self.port}",
                "OLLAMA_MODELS": str(download_dir / self.model_download_dir_name),
            }
        )
        server = Command(
            name="ollama-serve", command=[ollama_cli, "serve"], env=new_env
        )
        server.start()
        download_model = Command(
            name="ollama-download-model",
            command=[ollama_cli, "pull", self.model],
            env=new_env,
            long_living=False,
        )
        download_model.start()
        download_model.wait_and_log()
        server.stop()
        return {self.model_artifact_key: str(download_dir.absolute())}

    def _to_run_command(
        self, context: PythonModelContext = None
    ) -> Union[List[str], Command]:
        local_model_path = "/root/models"
        if context is not None:
            local_model_path = context.artifacts.get(self.model_artifact_key)
            set_full_permissions(str(local_model_path))
        ollama_root_dir = Path(local_model_path)
        bin_path = ollama_root_dir / "bin/ollama"
        new_env = os.environ.copy()
        new_env.update(
            {
                "OLLAMA_HOST": f"{self.host}:{self.port}",
                "OLLAMA_MODELS": str(ollama_root_dir / self.model_download_dir_name),
            }
        )
        return Command(name="ollama-serve", command=[bin_path, "serve"], env=new_env)

    def engine_pip_reqs(self, **kwargs) -> Dict[str, str]:
        return {}

    def supported_model_architectures(self) -> List[str]:
        return []


class OllamaEngineProcess(EngineProcess):

    @property
    def engine_name(self) -> str:
        return "ollama-engine"

    def health_check(self) -> bool:
        try:
            resp = self.server_http_client.get("/")
            return resp.status_code == 200
        except Exception as e:
            LOGGER.info(
                f"Health check failed with error {e}; server may not be up yet or crashed;"
            )
            return False
