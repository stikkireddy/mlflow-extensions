from mlflow_extensions.serving.compat import inject_mlflow_openai_compat_client

MODULE = "langchain_openai"
INSTALL = "langchain-openai"

try:
    from langchain_openai import ChatOpenAI, OpenAI
except ImportError as e:
    print(
        f"Error importing {MODULE} module please run "
        f"pip install {INSTALL} or upgrade the sdk by running pip install {INSTALL} --upgrade"
    )

OpenAI = inject_mlflow_openai_compat_client(use_sync=True, use_async=True)(OpenAI)
ChatOpenAI = inject_mlflow_openai_compat_client(use_sync=True, use_async=True)(
    ChatOpenAI
)
