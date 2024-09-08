import warnings

warnings.warn(
    "The module `mlflow_extensions.serving.fixtures` is deprecated. Please use `mlflow_extensions.testing.fixures` instead.",
    DeprecationWarning,
)
from mlflow_extensions.testing.fixures import LocalTestServer  # noqa
