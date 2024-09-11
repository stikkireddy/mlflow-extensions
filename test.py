import logging
import sys
from typing import List
import os

import structlog

from mlflow_extensions.log import Logger, LogLevel, get_logger, initialize_logging
from mlflow_extensions.log.handlers import (
    rotating_volume_handler,
)

stdout_handler: logging.Handler = logging.StreamHandler(sys.stdout)
volume_handler: logging.Handler = rotating_volume_handler(
    "test.log", "/Volumes/nfleming/default/logs", max_bytes=10,
)
initialize_logging(level=LogLevel.DEBUG, handlers=[stdout_handler, volume_handler])

logger: Logger = get_logger()
logger.debug("This is the message", foo="bar", baz="qux")
logger.info("This is the message", foo="bar", baz="qux")
logger.warning("This is the message", foo="bar", baz="qux")
logger.critical("This is the message", foo="bar", baz="qux")
logger.critical("This is the message", foo="bar", baz="qux")
logger.critical("This is the message", foo="bar", baz="qux")
