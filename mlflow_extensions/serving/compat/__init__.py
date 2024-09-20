import functools
from functools import wraps
from typing import Optional, Type
from urllib.parse import urlparse

import httpx
from httpx import URL, AsyncClient, Client, Request, Response

from mlflow_extensions.serving.serde_v2 import MlflowPyfuncHttpxSerializer


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


def get_ezdeploy_lite_openai_url(
    model_deployment_name: str,
    databricks_host: str = None,
    databricks_token: str = None,
    prefix: str = None,
):
    from mlflow_extensions.databricks.deploy.ez_deploy_lite import (
        EZ_DEPLOY_LITE_PREFIX,
        EzDeployLiteManager,
    )

    edlm = EzDeployLiteManager(
        databricks_host=databricks_host,
        databricks_token=databricks_token,
        prefix=prefix or EZ_DEPLOY_LITE_PREFIX,
    )
    return edlm.get_openai_url(model_deployment_name)


class BaseCustomMLFlowHttpClient:
    def __init__(
        self,
        *,
        endpoint_url: str,
        token: Optional[str] = None,
        timeout: int = 30,
        requires_openai_compat: bool = True,
    ):
        if validate_url_token(endpoint_url, token) is False:
            raise ValueError(
                "You must provide a token unless the endpoint is localhost or 0.0.0.0"
            )

        headers = {
            "Content-Type": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        self._custom_provided_base_path = urlparse(endpoint_url).path.rstrip("/")
        base_url = build_endpoint_url(endpoint_url)
        self._timeout = timeout
        self._http_client_args = {
            "base_url": base_url,
            "headers": headers,
            "timeout": timeout,
        }
        self.requires_openai_compat = requires_openai_compat

    def _prepare_request(self, request: Request) -> dict:
        url = request.url
        if isinstance(request.url, str):
            url = httpx.URL(request.url)
        path_to_request = url.path.replace(self._custom_provided_base_path, "")
        return {
            "inputs": [
                MlflowPyfuncHttpxSerializer.serialize_request(
                    request,
                    path_to_request,
                    requires_openai_compat=self.requires_openai_compat,
                )
            ]
        }

    @staticmethod
    def _process_response(response_json: dict, orig_request: Request) -> Response:
        try:
            prediction = response_json["predictions"][0]
        except KeyError:
            raise ValueError(
                f"Invalid response from server: missing 'predictions' key; response_json={response_json}"
            )
        return MlflowPyfuncHttpxSerializer.deserialize_response(
            prediction, orig_request
        )


class CustomMLFlowHttpClient(BaseCustomMLFlowHttpClient, Client):
    def __init__(self, **kwargs):
        BaseCustomMLFlowHttpClient.__init__(self, **kwargs)
        Client.__init__(self)
        self._http_client = httpx.Client(**self._http_client_args)

    def send(self, request: Request, **kwargs) -> Response:
        inputs = self._prepare_request(request)
        response = self._http_client.post("/invocations", json=inputs)
        return self._process_response(response.json(), request)


class AsyncCustomMLFlowHttpClient(BaseCustomMLFlowHttpClient, AsyncClient):
    def __init__(self, **kwargs):
        BaseCustomMLFlowHttpClient.__init__(self, **kwargs)
        AsyncClient.__init__(self)
        self._http_client = httpx.AsyncClient(**self._http_client_args)

    async def send(self, request: Request, **kwargs) -> Response:
        inputs = self._prepare_request(request)
        response = await self._http_client.post("/invocations", json=inputs)
        return self._process_response(response.json(), request)


def inject_mlflow_openai_compat_client(
    use_sync: Optional[bool] = None,
    use_async: Optional[bool] = None,
    sync_client_attribute: str = "http_client",
    async_client_attribute: str = "async_client",
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
                additional_client_kwargs[sync_client_attribute] = (
                    CustomMLFlowHttpClient(
                        endpoint_url=base_url, token=api_key, timeout=timeout
                    )
                )

            if use_async is True:
                additional_client_kwargs[async_client_attribute] = (
                    AsyncCustomMLFlowHttpClient(
                        endpoint_url=base_url, token=api_key, timeout=timeout
                    )
                )

            kwargs.update(additional_client_kwargs)
            original_init(self, **kwargs)

        cls.__init__ = new_init
        return cls

    return decorator


CustomServerClient = functools.partial(  # noqa
    CustomMLFlowHttpClient, requires_openai_compat=False
)
CustomServerClient: Type[BaseCustomMLFlowHttpClient]
CustomServerAsyncClient = functools.partial(  # noqa
    AsyncCustomMLFlowHttpClient, requires_openai_compat=False
)
CustomServerAsyncClient: Type[BaseCustomMLFlowHttpClient]
