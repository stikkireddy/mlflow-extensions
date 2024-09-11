from mlflow_extensions.log.handlers import rotating_volume_handler
from mlflow_extensions.log import initialize_logging, LogLevel, get_logger, Logger
import logging
import sys

volume_handler: logging.Handler = rotating_volume_handler("test.log", "/Volumes/foo/bar/baz")
stdout_handler: logging.Handler = logging.StreamHandler(sys.stdout)
initialize_logging(level=LogLevel.INFO, handlers=[volume_handler, stdout_handler])
logger: Logger = get_logger()
logger.info("Test", foo="bar", bar="baz", baz="qui")
