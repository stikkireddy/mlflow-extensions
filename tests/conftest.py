import os
import sys
import warnings
from pathlib import Path
from textwrap import dedent

import pytest
from dotenv import find_dotenv, load_dotenv

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["PYARROW_IGNORE_TIMEZONE"] = str(1)
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

src_dir: Path = Path(__file__).parents[1]
sys.path.insert(0, str(src_dir.resolve()))

env_path: str = find_dotenv()
load_dotenv(env_path)


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
