from mlflow_extensions.serving.compat import inject_mlflow_openai_compat_client

MODULE = "sglang"
INSTALL = "sglang"

try:
    from sglang import OpenAI
except ImportError as e:
    print(f"Error importing {MODULE} module please run "
          f"pip install {INSTALL} or upgrade the sdk by running pip install {INSTALL} --upgrade")

OpenAI = inject_mlflow_openai_compat_client(use_sync=True)(OpenAI)
