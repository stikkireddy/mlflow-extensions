from mlflow_extensions.serving.compat import inject_mlflow_openai_compat_client

MODULE = "openai"
INSTALL = "openai"

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError as e:
    print(
        f"Error importing {MODULE} module please run "
        f"pip install {INSTALL} or upgrade the sdk by running pip install {INSTALL} --upgrade"
    )

OpenAI = inject_mlflow_openai_compat_client(use_sync=True)(OpenAI)
AsyncOpenAI = inject_mlflow_openai_compat_client(use_async=True)(AsyncOpenAI)
