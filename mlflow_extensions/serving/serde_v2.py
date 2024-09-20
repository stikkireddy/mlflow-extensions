import json
from base64 import b64decode, b64encode
from io import BytesIO
from typing import Any, Dict

import httpx
from httpx import URL, Request


class RequestSerdeKeys:
    METHOD = "method"
    URL_PATH = "url_path"
    HEADERS = "headers"
    BODY = "body"
    BODY_TYPE = "body_type"
    REQUIRES_OPENAI_COMPAT = "requires_openai_compat"


class ResponseSerdeKeys:
    STATUS_CODE = "status_code"
    HEADERS = "headers"
    CONTENT = "content"
    CONTENT_TYPE = "content_type"


class SerdeContentTypes:
    MULTIPART = "multipart"
    TEXT = "text"
    STREAM = "stream"
    NONE = "none"


class MlflowPyfuncHttpxSerializer:
    @staticmethod
    def serialize_request(
        request: Request, url_path_to_request: str, requires_openai_compat: bool = True
    ):
        headers = dict(request.headers)

        if headers["content-type"].startswith("multipart/form-data"):
            body_type = SerdeContentTypes.MULTIPART
            body = request.read().decode("utf-8")
        elif isinstance(request.content, str):
            body = request.content
            body_type = SerdeContentTypes.TEXT
        elif isinstance(request.content, bytes):
            if request.headers.get("content-type", "").startswith(
                "application/json"
            ) or request.headers.get("content-type", "").startswith("text/plain"):
                body = request.content.decode("utf-8")
                body_type = SerdeContentTypes.TEXT
            else:
                body = b64encode(request.content).decode("utf-8")
                body_type = SerdeContentTypes.STREAM
        elif request.content is None:
            body = None
            body_type = SerdeContentTypes.NONE
        else:
            body = request.read()
            body = b64encode(body).decode("utf-8")
            body_type = SerdeContentTypes.STREAM

        serialized = {
            RequestSerdeKeys.METHOD: request.method,
            RequestSerdeKeys.URL_PATH: url_path_to_request,
            RequestSerdeKeys.HEADERS: {
                "content-type": headers.get("content-type", "text/plain")
            },
            RequestSerdeKeys.BODY: body,
            RequestSerdeKeys.BODY_TYPE: body_type,
            RequestSerdeKeys.REQUIRES_OPENAI_COMPAT: requires_openai_compat,
        }
        return json.dumps(serialized)

    @staticmethod
    def deserialize_request(
        serialized_request: str, *, openai_base_url: URL, server_base_url: URL
    ):
        req_data = json.loads(serialized_request)

        if req_data[RequestSerdeKeys.BODY_TYPE] == SerdeContentTypes.STREAM:
            content = BytesIO(b64decode(req_data[RequestSerdeKeys.BODY]))
        elif req_data[RequestSerdeKeys.BODY_TYPE] == SerdeContentTypes.MULTIPART:
            content = BytesIO(req_data[RequestSerdeKeys.BODY].encode("utf-8"))
        elif req_data[RequestSerdeKeys.BODY_TYPE] == SerdeContentTypes.NONE:
            content = None
        else:
            content = req_data[RequestSerdeKeys.BODY]

        if req_data[RequestSerdeKeys.REQUIRES_OPENAI_COMPAT] is True:
            base_url = openai_base_url
        else:
            base_url = server_base_url
        full_url = base_url.join(req_data[RequestSerdeKeys.URL_PATH].lstrip("/"))

        return httpx.Request(
            method=req_data[RequestSerdeKeys.METHOD],
            url=full_url,
            headers=req_data[RequestSerdeKeys.HEADERS],
            content=content,
        )

    @staticmethod
    def serialize_response(response):
        headers = dict(response.headers)

        # Read and encode the entire content
        if headers.get("content-type", "").startswith(
            "application/json"
        ) or headers.get("content-type", "").startswith("text/plain"):
            content = response.read().decode("utf-8")
            content_type = SerdeContentTypes.TEXT
        else:
            content = response.read()
            content = b64encode(content).decode("utf-8")
            content_type = SerdeContentTypes.STREAM

        serialized = {
            ResponseSerdeKeys.STATUS_CODE: response.status_code,
            ResponseSerdeKeys.HEADERS: headers,
            ResponseSerdeKeys.CONTENT: content,
            ResponseSerdeKeys.CONTENT_TYPE: content_type,
        }
        return json.dumps(serialized)

    @staticmethod
    def deserialize_response(serialized_response, orig_request: Request = None):
        resp_data = json.loads(serialized_response)

        if resp_data[ResponseSerdeKeys.CONTENT_TYPE] == SerdeContentTypes.STREAM:
            content = b64decode(resp_data[ResponseSerdeKeys.CONTENT])
        else:
            content = resp_data[ResponseSerdeKeys.CONTENT]

        return httpx.Response(
            status_code=resp_data[ResponseSerdeKeys.STATUS_CODE],
            headers=resp_data[ResponseSerdeKeys.HEADERS],
            content=content,
            request=orig_request,
        )


def make_error_response(
    *,
    original_request: Request,
    error_message: str,
    error_type: str,
    error_details: Dict[str, Any],
    status_code: int = 500
):
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=json.dumps(
            {
                "error": error_message,
                "error_type": error_type,
                "error_details": error_details,
                "original_request": MlflowPyfuncHttpxSerializer.serialize_request(
                    original_request, url_path_to_request=original_request.url.path
                ),
            }
        ),
    )
