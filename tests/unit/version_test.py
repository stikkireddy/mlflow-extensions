import re

import pytest

from mlflow_extensions.version import get_mlflow_extensions_version


@pytest.mark.unit
def test_should_return_correct_version() -> None:
    version: str = get_mlflow_extensions_version()
    assert version is not None

    regex: str = r"^(\d+\.\d+)(\.dev\d+)?\+g([0-9a-f]+)\.d(\d{8})$"

    match: re.Match = re.match(regex, version)
    assert version is not None and match is not None
    if match:
        print("Base Version:", match.group(1))
        print("Development Version:", match.group(2))
        print("Git Commit Hash:", match.group(3))
        print("Build Date:", match.group(4))
