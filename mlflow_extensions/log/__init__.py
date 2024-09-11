import inspect
import linecache
import logging
import os
import sys
from enum import IntEnum
from types import FrameType
from typing import Any, Callable, Dict, List, Optional

import structlog
import structlog.stdlib
from pythonjsonlogger import jsonlogger
from structlog.stdlib import BoundLogger
from structlog.types import EventDict, WrappedLogger

Logger = BoundLogger


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


def initialize_logging(
    level: LogLevel = LogLevel.INFO, handlers: List[logging.Handler] = []
) -> None:

    json_formatter: jsonlogger.JsonFormatter = jsonlogger.JsonFormatter()

    for handler in handlers:
        handler.setFormatter(json_formatter)
        handler.setLevel(level)

    logging.basicConfig(level=level, format="%(message)s", handlers=handlers)

    def filter_by_level(logger: Logger, name: str, event_dict: EventDict) -> EventDict:
        event_log_level: LogLevel = LogLevel.from_string(name)
        if event_log_level >= level:
            return event_dict
        else:
            raise structlog.DropEvent

    structlog.configure_once(
        processors=[
            filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
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
