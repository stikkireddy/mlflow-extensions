import logging
import os
import sys
from typing import Callable, List

import pytest
import structlog

from mlflow_extensions.log import Logger, LogLevel, get_logger, initialize_logging
from mlflow_extensions.log.handlers import (
    FileRotator,
    VolumeRotator,
    _full_volume_name_to_path,
    _get_databricks_host_creds,
    create_rotator,
    rotating_volume_handler,
)


def test_full_volume_name_to_path() -> None:
    full_volume_name: str = "foo.bar.baz"
    expected_path: str = "/Volumes/foo/bar/baz"
    assert _full_volume_name_to_path(full_volume_name) == expected_path


def test_intialize_logging_with_defaults() -> None:
    initialize_logging()
    assert structlog.is_configured()


def test_get_logger_should_return_default_log_level_and_name() -> None:
    volume_handler: logging.Handler = rotating_volume_handler(
        "/tmp/test.log", "/Volumes/foo/bar/baz"
    )
    stdout_handler: logging.Handler = logging.StreamHandler(sys.stdout)
    initialize_logging(level=LogLevel.DEBUG, handlers=[volume_handler, stdout_handler])
    logger: Logger = get_logger()
    logger.info("This is the message", foo="bar", baz="qux")


def test_create_rotator_with_no_volume() -> None:
    assert isinstance(create_rotator(None, None, None), FileRotator)


def test_create_rotator_with_volume_path() -> None:
    assert isinstance(
        create_rotator("/Volumes/nfleming/default/logs", None, None), VolumeRotator
    )


def test_create_rotator_with_volume_path_env_vars() -> None:
    os.environ["LOGGING_VOLUME"] = "nfleming.default.logs"
    os.environ["LOGGING_MODEL_NAME"] = "my_model"
    os.environ["LOGGING_ENDPOINT_ID"] = "my_endpoint"
    os.environ["LOGGING_RUN_ID"] = "my_run"

    assert isinstance(create_rotator(None, None, None), VolumeRotator)


@pytest.mark.skipif(
    "DATABRICKS_HOST" not in os.environ,
    reason="Missing environment variable: DATABRICKS_HOST",
)
@pytest.mark.skipif(
    "DATABRICKS_TOKEN" not in os.environ,
    reason="Missing environment variable: DATABRICKS_TOKEN",
)
def test_databricks_host_creds() -> None:
    host, token = _get_databricks_host_creds(None, None)
    assert host is not None
    assert token is not None
