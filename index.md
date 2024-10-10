---
hide:
  - navigation
---

# Mlflow Extensions

The goal of this project is to make deploying any large language model, or multi modal large language models
a simple three-step process.

1. Download the model from hf or any other source.
2. Register the model with mlflow.
3. Deploy the model using the mlflow serving infrastructure. (e.g. Databricks)

## Framework Support / Roadmap

- [x] [vLLM](https://github.com/vllm-project/vllm){:target="_blank"}
- [x] [SGLang](https://github.com/sgl-project/sglang){:target="_blank"}
- [ ] [Ray Serving [WIP]](https://github.com/ray-project/ray){:target="_blank"}

This project will take those optimized model serving frameworks and deploy them to the
following deployment targets.

## Deployment Clouds

- [x] AWS
- [x] Azure
- [x] GCP

## Deployment Targets

- [x] Databricks Model Serving
- [x] Databricks Job Cluster
- [ ] Databricks Interactive Clusters

## Deployment Modes

- [x] EzDeployLite will ship a prebuilt configuration to databricks jobs. (dev/testing)
- [x] EzDeploy will ship a prebuilt configuration to databricks model serving. (production)

## Disclaimer

mlflow-extensions is not developed, endorsed not supported by Databricks. It is provided as-is; no warranty is derived
from using this package.
For more details, please refer to the license.