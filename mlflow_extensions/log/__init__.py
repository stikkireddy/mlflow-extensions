import inspect
import logging
import os
import socket
from enum import IntEnum
from types import FrameType
from typing import Any, Callable, Dict, List, Optional

import structlog
import structlog.stdlib
from pythonjsonlogger import jsonlogger
from structlog.stdlib import BoundLogger
from structlog.types import EventDict, WrappedLogger

from mlflow_extensions.version import get_mlflow_extensions_version

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

    logging.shutdown()
    logging.basicConfig(level=level, format="%(message)s", handlers=handlers)

    def filter_by_level(logger: Logger, name: str, event_dict: EventDict) -> EventDict:
        event_log_level: LogLevel = LogLevel.from_string(name)
        if event_log_level >= level:
            return event_dict
        else:
            raise structlog.DropEvent

    version: str = get_mlflow_extensions_version()

    def add_library_version(
        logger: Logger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        event_dict["version"] = version
        return event_dict

    hostname: str = socket.gethostname()

    def add_hostname(
        logger: Logger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        event_dict["host"] = hostname
        return event_dict

    ip: str = socket.gethostbyname(hostname)

    def add_ip_address(
        logger: Logger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        event_dict["ip"] = socket.gethostbyname(socket.gethostname())
        return event_dict

    model_name: str = os.environ.get("LOGGING_MODEL_NAME")
    endpoint_id: str = os.environ.get("LOGGING_ENDPOINT_ID")
    run_id: str = os.environ.get("LOGGING_RUN_ID")

    def add_model_info(
        logger: Logger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        if model_name is not None:
            event_dict["model"] = model_name
        if endpoint_id is not None:
            event_dict["endpoint_id"] = endpoint_id
        if run_id is not None:
            event_dict["run_id"] = run_id
        return event_dict

    structlog.configure_once(
        processors=[
            filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            add_hostname,
            add_ip_address,
            add_library_version,
            add_model_info,
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
