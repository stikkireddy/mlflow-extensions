import functools
import inspect
import logging
import os
import socket
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from types import FrameType
from typing import Any, Callable, Dict, List, Optional, Union

import structlog
import structlog.stdlib
from pythonjsonlogger import jsonlogger
from structlog.stdlib import BoundLogger
from structlog.types import EventDict, WrappedLogger

from mlflow_extensions.log.handlers import rotating_volume_handler
from mlflow_extensions.version import get_mlflow_extensions_version

Logger = BoundLogger

DEFAULT_MAX_BYTES: int = 10 * 1024 * 1024
DEFAULT_BACKUP_COUNT: int = 5


class LogLevel(IntEnum):
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    @classmethod
    def from_int(cls, value: int):
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"No LogLevel with integer value {value}")

    @classmethod
    def from_string(cls, name: str):
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"No LogLevel with name {name}")

    @classmethod
    def to_level(cls, level: Union["LogLevel", str, int]) -> "LogLevel":
        if isinstance(level, LogLevel):
            return level
        elif isinstance(level, str):
            return cls.from_string(level)
        elif isinstance(level, int):
            return cls.from_int(level)
        else:
            raise ValueError(f"Invalid log level: {level}")


@dataclass
class LogConfig:
    filename: Optional[str] = None
    level: Union[LogLevel, int, str] = LogLevel.INFO
    archive_path: Optional[str] = None
    databricks_host: Optional[str] = None
    databricks_token: Optional[str] = None
    when: str = "m"
    interval: int = 5
    max_bytes: int = DEFAULT_MAX_BYTES
    backup_count: int = DEFAULT_BACKUP_COUNT
    encoding: Optional[str] = None
    delay: bool = False
    utc: bool = False
    at_time = None
    errors = None
    additional_vars: Dict[str, Any] = field(default_factory=dict)


def get_logger(name: Optional[str] = None) -> Logger:
    """
    Returns a logger with the specified name.

    Parameters:
        name (Optional[str]): The name of the logger. If not provided, the name will be determined based on the caller's module and class (if applicable).

    Returns:
        Logger: The logger object.

    Example:
        logger: Logger = get_logger()
    """

    if name is None:
        frame: Optional[FrameType] = inspect.currentframe()
        caller_frame: Optional[FrameType] = frame.f_back
        module: Dict[str, Any] = caller_frame.f_globals["__name__"]

        # Check if it's a method inside a class and get the class name
        if "self" in caller_frame.f_locals:
            class_name: str = caller_frame.f_locals["self"].__class__.__name__
            name = f"{module}.{class_name}"
        else:
            name = module

    logger: Logger = structlog.stdlib.get_logger(name)

    return logger


def initialize_logging(config: LogConfig) -> None:

    stdout_handler: logging.Handler = logging.StreamHandler(sys.stdout)
    handlers: List[logging.Handler] = [stdout_handler]

    if config.filename is not None:
        volume_handler: logging.Handler = rotating_volume_handler(
            filename=config.filename,
            archive_path=config.archive_path,
            max_bytes=config.max_bytes,
            backup_count=config.backup_count,
            databricks_host=config.databricks_host,
            databricks_token=config.databricks_token,
            when=config.when,
            interval=config.interval,
        )
        handlers.append(volume_handler)

    level: LogLevel = LogLevel.to_level(config.level)
    json_formatter: jsonlogger.JsonFormatter = jsonlogger.JsonFormatter()

    for handler in handlers:
        handler.setFormatter(json_formatter)
        handler.setLevel(level)

    logging.shutdown()
    logging.basicConfig(level=level, format="%(message)s", handlers=handlers)

    def filter_by_level(logger: Logger, name: str, event_dict: EventDict) -> EventDict:
        event_log_level: LogLevel = LogLevel.from_string(name)
        if event_log_level >= level:
            return event_dict
        else:
            raise structlog.DropEvent

    def add_additional_vars(
        logger: Logger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        for key, value in config.additional_vars.items():
            event_dict[key] = value
        return event_dict

    structlog.configure(
        processors=[
            filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            add_additional_vars,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.ExceptionRenderer(),
            structlog.processors.UnicodeDecoder(),
            # structlog.processors.CallsiteParameterAdder(
            #     {
            #         structlog.processors.CallsiteParameter.FILENAME,
            #         structlog.processors.CallsiteParameter.FUNC_NAME,
            #         structlog.processors.CallsiteParameter.LINENO,
            #     }
            # ),
            # structlog.processors.JSONRenderer(),
            structlog.stdlib.render_to_log_kwargs,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def log_around(_func=None, *, logger: Union[Logger, logging.Logger] = None):
    def decorator_log(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__name__)
            args_repr: List[str] = [repr(a) for a in args]
            kwargs_repr: List[str] = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature: str = ", ".join(args_repr + kwargs_repr)
            logger.debug(f"entry: {func.__name__}", signature=signature)
            try:
                result = func(*args, **kwargs)
                logger.debug(f"exit: {func.__name__}", result=result)
                return result
            except Exception as e:
                logger.exception(
                    f"Exception raised in {func.__name__}. exception: {str(e)}"
                )
                raise e

        return wrapper

    if _func is None:
        return decorator_log
    else:
        return decorator_log(_func)
