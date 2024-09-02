from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlflow-extensions",
    author="Sri Tikkireddy",
    author_email="sri.tikkireddy@databricks.com",
    description="Extensions for mlflow to make the devloop better for custom models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stikkireddy/mlflow-extensions",
    packages=find_packages(),
    install_requires=["httpx", "huggingface-hub", "filelock", "mlflow-skinny[databricks]", "psutil", "databricks-sdk"],
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
