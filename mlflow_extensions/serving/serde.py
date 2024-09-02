import abc
import json
from dataclasses import dataclass
from typing import List, Union


class MessageProto(abc.ABC):
    # first item in the array is the version

    def is_version(self, data: List[str]) -> bool:
        if len(data) > 0:
            return data[0] == self.version()
        return False

    def version(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def _serialize(self) -> List[str]:
        pass

    def serialize(self) -> List[str]:
        return [self.version(), *self._serialize()]

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, data: List[str]) -> "MessageProto":
        pass


@dataclass(kw_only=True)
class RequestMessageV1(MessageProto):
    request_path: str  # 1
    method: str  # 2
    timeout: int  # 3
    payload: Union[str, dict]  # 4

    def __post_init__(self):
        if isinstance(self.payload, dict):
            self.payload = json.dumps(self.payload)

    def _serialize(self) -> List[str]:
        return [self.request_path, self.method, str(self.timeout), self.payload]

    @classmethod
    def deserialize(cls, data: List[str]) -> "RequestMessageV1":
        return cls(
            request_path=data[1], method=data[2], timeout=int(data[3]), payload=data[4]
        )


@dataclass(frozen=True, kw_only=True)
class ResponseMessageV1(MessageProto):
    request_method: str
    request_timeout: int
    response_data: str
    response_status_code: int
    response_content_type: str

    def _serialize(self) -> List[str]:
        return [
            self.request_method,
            str(self.request_timeout),
            self.response_data,
            str(self.response_status_code),
            self.response_content_type,
        ]

    @classmethod
    def deserialize(cls, data: List[str]) -> "ResponseMessageV1":
        return cls(
            request_method=data[1],
            request_timeout=int(data[2]),
            response_data=data[3],
            response_status_code=int(data[4]),
            response_content_type=data[5],
        )
