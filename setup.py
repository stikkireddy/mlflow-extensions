from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read().splitlines()

setup(
    name="mlflow-extensions",
    author="Sri Tikkireddy",
    author_email="sri.tikkireddy@databricks.com",
    description="Extensions for mlflow to make the devloop better for custom models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stikkireddy/mlflow-extensions",
    packages=find_packages(),
    install_requires=install_requires,
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
