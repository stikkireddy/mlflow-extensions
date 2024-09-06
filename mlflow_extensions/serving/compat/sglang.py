import functools
import importlib
import sys
from typing import Optional

from mlflow_extensions.serving.compat import (
    CustomMLFlowHttpClient,
    inject_mlflow_openai_compat_client,
)

MODULE = "sglang"
INSTALL = "sglang"

try:
    from sglang import OpenAI
except ImportError as e:
    print(
        f"Error importing {MODULE} module please run "
        f"pip install {INSTALL} or upgrade the sdk by running pip install {INSTALL} --upgrade"
    )

OpenAI = inject_mlflow_openai_compat_client(use_sync=True)(OpenAI)


@functools.lru_cache(maxsize=32)
def get_client(url: str, api_key: str) -> "CustomMLFlowHttpClient":
    return CustomMLFlowHttpClient(
        endpoint_url=url, token=api_key, timeout=30, requires_openai_compat=False
    )


def RuntimeEndpoint(
    base_url: str,
    api_key: Optional[str] = None,
    verify: Optional[str] = None,
):
    import httpx
    from sglang import RuntimeEndpoint

    def patched_http_request(url, json=None, stream=False, api_key=None, verify=None):
        """A patched version of the httpx request to modify sglang interaction"""
        headers = {"Content-Type": "application/json; charset=utf-8"}

        parsed_url = httpx.URL(url)
        client = get_client(base_url, api_key)
        resp = client.send(
            httpx.Request(
                method="POST" if json is not None else "GET",
                json=json,
                headers=headers,
                url=parsed_url,
            )
        )
        return resp

    sys.modules["sglang.utils"].__dict__["http_request"] = patched_http_request
    importlib.reload(sys.modules["sglang.lang.backend.runtime_endpoint"])
    return RuntimeEndpoint(base_url, api_key, verify)
