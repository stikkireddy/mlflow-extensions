from functools import wraps
from typing import Optional
from urllib.parse import urlparse

import httpx
from httpx import Client, Request, Response, AsyncClient

from mlflow_extensions.serving.serde import RequestMessageV1, ResponseMessageV1


def is_local(url: str) -> bool:
    return "0.0.0.0" in url or "localhost" in url or "127.0.0.1" in url


def validate_url_token(url: str, token: Optional[str] = None) -> bool:
    if is_local(url) is False and token is None:
        return False
    return True


def build_endpoint_url(url: str) -> str:
    normalized_url = url.rstrip("/")
    if normalized_url.endswith("/invocations"):
        normalized_url = normalized_url.replace("/invocations", "")
    if is_local(normalized_url):
        return normalized_url.rstrip("/")
    return normalized_url


class CustomMLFlowHttpClient(Client):

    def __init__(self, *, endpoint_url: str,
                 token: Optional[str] = None,
                 timeout: int = 30):
        super().__init__()
        if validate_url_token(endpoint_url, token) is False:
            raise ValueError("You must provide a token unless the endpoint is localhost or 0.0.0.0")
        headers = {
            'Content-Type': 'application/json',
        }
        if token:
            headers['Authorization'] = f"Bearer {token}"
        self._custom_provided_base_path = urlparse(endpoint_url).path.rstrip("/")
        base_url = build_endpoint_url(endpoint_url)
        self._http_client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        self._timeout = timeout

    def send(self, request: Request, **kwargs) -> Response:
        openai_path_to_request = request.url.path.replace(self._custom_provided_base_path, "")
        content = request.content.decode("utf-8")
        req = RequestMessageV1(
            request_path=openai_path_to_request,
            payload=content,
            method=request.method,
            timeout=self._timeout,
        )
        inputs = {"inputs": [req.serialize()]}
        response = self._http_client.post("/invocations", json=inputs)
        resp_data = ResponseMessageV1.deserialize(response.json()["predictions"][0])
        return Response(
            request=response.request,
            status_code=resp_data.response_status_code,
            headers={"Content-Type": resp_data.response_content_type},
            content=resp_data.response_data
        )


class AsyncCustomMLFlowHttpClient(AsyncClient):

    def __init__(self, *, endpoint_url: str,
                 token: Optional[str] = None,
                 timeout: int = 30):
        super().__init__()
        if validate_url_token(endpoint_url, token) is False:
            raise ValueError("You must provide a token unless the endpoint is localhost or 0.0.0.0")
        headers = {
            'Content-Type': 'application/json',
        }
        if token:
            headers['Authorization'] = f"Bearer {token}"
        self._custom_provided_base_path = urlparse(endpoint_url).path.rstrip("/")
        base_url = build_endpoint_url(endpoint_url)
        self._http_client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout)
        self._timeout = timeout

    async def send(self, request: Request, **kwargs) -> Response:
        openai_path_to_request = request.url.path.replace(self._custom_provided_base_path, "")
        content = request.content.decode("utf-8")
        req = RequestMessageV1(
            request_path=openai_path_to_request,
            payload=content,
            method=request.method,
            timeout=self._timeout,
        )
        content = {"inputs": [req.serialize()]}
        response = await self._http_client.post("/invocations", json=content)
        resp_data = ResponseMessageV1.deserialize(response.json()["predictions"][0])
        return Response(
            request=response.request,
            status_code=resp_data.response_status_code,
            headers={"Content-Type": resp_data.response_content_type},
            content=resp_data.response_data
        )


def inject_mlflow_openai_compat_client(
        use_sync: Optional[bool] = None,
        use_async: Optional[bool] = None,
        sync_client_attribute: str = "http_client",
        async_client_attribute: str = "async_client"
):
    def decorator(cls):
        original_init = cls.__init__

        @wraps(original_init)
        def new_init(self, **kwargs):
            if kwargs.get("streaming") is True:
                raise ValueError("Streaming is not supported with this adapter")
            assert "base_url" in kwargs, "You must provide a base_url"
            base_url = kwargs.get("base_url")
            api_key = kwargs.get("api_key")
            timeout = int(kwargs.get("timeout", 30))
            if validate_url_token(base_url, api_key) is False:
                raise ValueError("You must provide an api_key")

            additional_client_kwargs = {}
            if use_sync is True:
                additional_client_kwargs[sync_client_attribute] = CustomMLFlowHttpClient(
                    endpoint_url=base_url,
                    token=api_key,
                    timeout=timeout
                )

            if use_async is True:
                additional_client_kwargs[async_client_attribute] = AsyncCustomMLFlowHttpClient(
                    endpoint_url=base_url,
                    token=api_key,
                    timeout=timeout
                )

            kwargs.update(additional_client_kwargs)
            original_init(self, **kwargs)

        cls.__init__ = new_init
        return cls

    return decorator
