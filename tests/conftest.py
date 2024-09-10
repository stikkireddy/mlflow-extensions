import os
import sys
import warnings
from pathlib import Path
from textwrap import dedent

import pytest
from databricks.sdk import WorkspaceClient
from dotenv import find_dotenv, load_dotenv
from mlflow import MlflowClient

src_dir: Path = Path(__file__).parents[1]
sys.path.insert(0, str(src_dir.resolve()))

env_path: str = find_dotenv()
load_dotenv(env_path)


@pytest.fixture
def mlflow_client() -> MlflowClient:
    return MlflowClient()


@pytest.fixture
def workspace_client() -> WorkspaceClient:
    return WorkspaceClient(
        host=os.environ.get("DATABRICKS_HOST"), token=os.environ.get("DATABRICKS_TOKEN")
    )


# Alias HF_TOKEN to HUGGINGFACEHUB_API_TOKEN
if any([e in os.environ for e in ["HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"]]):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get(
        "HUGGINGFACEHUB_API_TOKEN", os.environ.get("HF_TOKEN")
    )


def warn_if_missing(*env_vars: str) -> None:
    has_missing: bool = False
    for env_var in env_vars:
        if env_var not in os.environ:
            has_missing = True
            warnings.warn(f"Missing required env var: {env_var}")
    if has_missing:
        warnings.warn(
            dedent(
                """
      Environmnt variables can be set from the shell or a .env file     
    """
            )
        )


warn_if_missing("DATABRICKS_HOST", "DATABRICKS_TOKEN")
