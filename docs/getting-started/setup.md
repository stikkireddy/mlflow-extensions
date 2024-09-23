# Setup

MLflow extensions deployments are split into two main parts, `EzDeployLite` and `EzDeploy`.
`EzDeployLite` is a lightweight deployment that is meant for development and testing. It runs on jobs service and uses
driver proxy. `EzDeploy` is a full deployment that is meant for production which gets deployed into model serving.
**EzDeployLite deploys to AWS, Azure and GCP.** **EzDeploy deploys to any regions supporting model serving with gpus.**

## 1: Requirements

1. You need access to create a job or a cluster with gpus T4, A10, A100 or H100.
2. You need access to a cluster to run the deployment (any compute, serverless, or non serverless)
3. Ability to download a model from huggingface or any other source.
4. Ability to install mlflow-extensions.

## 2: Installation

```python
%pip install mlflow-extensions
dbutils.library.restartPython()
```

The previous command installs the mlflow-extensions library. You can then import the library and use it in your code.
Run through the EzDeployLite or EzDeploy examples to see how to use the library for deploying models.