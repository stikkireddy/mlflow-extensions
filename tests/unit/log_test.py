import logging
import os
import re
import sys
from datetime import datetime
from typing import Generator, List

import pytest
import structlog
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType
from databricks.sdk.service.files import DirectoryEntry

from mlflow_extensions.log import Logger, LogLevel, get_logger, initialize_logging
from mlflow_extensions.log.handlers import (
    full_volume_name_to_path,
    rotating_volume_handler,
)


def test_full_volume_name_to_path() -> None:
    full_volume_name: str = "foo.bar.baz"
    expected_path: str = "/Volumes/foo/bar/baz"
    assert full_volume_name_to_path(full_volume_name) == expected_path


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
    logger.info("Test")
