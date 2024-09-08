import os
import sys
from pathlib import Path

import pytest

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["PYARROW_IGNORE_TIMEZONE"] = str(1)
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

src_dir: Path = Path(__file__).parents[1]
sys.path.insert(0, str(src_dir.resolve()))

