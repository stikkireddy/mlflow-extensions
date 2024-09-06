from importlib.metadata import PackageNotFoundError, version


def get_mlflow_extensions_version():
    try:
        return version("mlflow-extensions")
    except PackageNotFoundError:
        # package is not installed
        return None
