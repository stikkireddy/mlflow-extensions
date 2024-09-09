import re
from re import Pattern

import pytest

from mlflow_extensions.version import get_mlflow_extensions_version


@pytest.mark.unit
def test_should_return_correct_version() -> None:
    version: str = get_mlflow_extensions_version()
    assert version is not None

    pep440_regex: Pattern = re.compile(
        r"""
    ^
    (?P<version>
        (0|[1-9]\d*)                # Major version
        (\.(0|[1-9]\d*))*           # Minor and Patch versions (optional, can have more)
    )
    (?P<pre>
        ((-|\.)(a|b|rc)             # Pre-release identifier (alpha, beta, rc)
        (0|[1-9]\d*))               # Pre-release number
    )?
    (?P<post>
        (\.post(0|[1-9]\d*))        # Post-release identifier
    )?
    (?P<dev>
        (\.dev(0|[1-9]\d*))         # Development release identifier
    )?
    (?P<local>
        (\+[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*)   # Local version identifier (optional)
    )?
    $
    """,
        re.VERBOSE,
    )

    match: re.Match = re.match(pep440_regex, version)
    assert version is not None and match is not None
    if match:
        print("Major/Minor:", match.group(1))
        print("Pre-release:", match.group(2))
        print("Post-release:", match.group(3))
        print("Local-Version:", match.group(4))
