import logging
import os
import sys
from typing import Callable, List

import pytest
import structlog

from mlflow_extensions.log import (
    LogConfig,
    Logger,
    LogLevel,
    get_logger,
    initialize_logging,
)
from mlflow_extensions.log.handlers import (
    FileRotator,
    VolumeRotator,
    _get_databricks_host_creds,
    create_rotator,
    full_volume_name_to_path,
    rotating_volume_handler,
)


def test_full_volume_name_to_path() -> None:
    full_volume_name: str = "foo.bar.baz"
    expected_path: str = "/Volumes/foo/bar/baz"
    assert full_volume_name_to_path(full_volume_name) == expected_path


def test_intialize_logging_with_defaults() -> None:
    initialize_logging(LogConfig(filename="test.log"))
    assert structlog.is_configured()


def test_get_basic_logger_should_be_filtered() -> None:
    initialize_logging(
        LogConfig(
            level=LogLevel.DEBUG,
            filename="/tmp/test.log",
        )
    )
    logger: Logger = logging.getLogger()
    logger.info("This is the message")


def test_get_logger_should_return_default_log_level_and_name() -> None:
    initialize_logging(
        LogConfig(
            level=LogLevel.DEBUG,
            filename="/tmp/test.log",
            archive_path="/Volumes/nfleming/main/logs/test",
        )
    )
    logger: Logger = get_logger()
    logger.info("This is the message", foo="bar", baz="qux")


def test_get_logger_should_return_default_log_level_and_name_no_volume() -> None:
    initialize_logging(LogConfig(level=LogLevel.DEBUG, filename="/tmp/test.log"))
    logger: Logger = get_logger()
    logger.info("This is the message", foo="bar", baz="qux")


def test_create_rotator_with_no_volume() -> None:
    assert isinstance(create_rotator(None, None, None), FileRotator)


@pytest.mark.skipif(
    "DATABRICKS_HOST" not in os.environ,
    reason="Missing environment variable: DATABRICKS_HOST",
)
@pytest.mark.skipif(
    "DATABRICKS_TOKEN" not in os.environ,
    reason="Missing environment variable: DATABRICKS_TOKEN",
)
def test_create_rotator_with_volume_path() -> None:
    assert isinstance(
        create_rotator("/Volumes/nfleming/default/logs", None, None), VolumeRotator
    )


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
