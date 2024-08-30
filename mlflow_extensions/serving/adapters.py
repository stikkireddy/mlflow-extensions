import json
import typing
from typing import Optional

import httpx
from httpx import Client, Request, Response

if typing.TYPE_CHECKING:
    from openai import OpenAI
    from langchain_openai import ChatOpenAI


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
        base_url = build_endpoint_url(endpoint_url)
        self._http_client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)

    def send(self, request: Request, **kwargs) -> Response:
        content = request.content.decode("utf-8")
        content = {"inputs": [content]}
        response = self._http_client.post("/invocations", json=content)
        resp_data = json.loads(response.json()["predictions"][0])
        response_content = resp_data["data"]
        response_status = resp_data["status"]
        return Response(
            request=response.request,
            status_code=response_status,
            headers={"Content-Type": "application/json"},
            content=response_content)


def OpenAIWrapper(*args, **kwargs) -> 'OpenAI':
    kwargs.pop("http_client", None)
    assert "base_url" in kwargs, "You must provide a base_url"
    base_url = kwargs["base_url"]
    api_key = kwargs.get("api_key")
    if validate_url_token(base_url, api_key) is False:
        raise ValueError("You must provide an api_key")

    from openai import OpenAI
    return OpenAI(*args, **kwargs, http_client=CustomMLFlowHttpClient(
        endpoint_url=base_url,
        token=api_key
    ))


def ChatOpenAIWrapper(**kwargs) -> 'ChatOpenAI':
    kwargs.pop("http_client", None)
    assert "base_url" in kwargs, "You must provide a base_url"
    base_url = kwargs["base_url"]
    api_key = kwargs.get("api_key")
    if validate_url_token(base_url, api_key) is False:
        raise ValueError("You must provide an api_key")

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(**kwargs, http_client=CustomMLFlowHttpClient(endpoint_url=base_url, token=api_key))
