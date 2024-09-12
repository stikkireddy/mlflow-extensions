import pytest


@pytest.mark.unit
def test_vllm_config():
    from mlflow_extensions.databricks.prebuilt.text.vllm import NUEXTRACT_CONFIG

    assert len(NUEXTRACT_CONFIG.engine_config.default_pip_reqs()) > 0


@pytest.mark.unit
def test_sglang_config():
    from mlflow_extensions.databricks.prebuilt.text.sglang import GEMMA_2_9B_IT_CONFIG

    pip_reqs = GEMMA_2_9B_IT_CONFIG.engine_config.default_pip_reqs()
    assert len(pip_reqs) > 0, "pip_reqs not found"
    assert (
        "--extra-index-url=https://flashinfer.ai/whl/cu121/torch2.4/" in pip_reqs
    ), "flashinfer extra index url not found"


@pytest.mark.unit
def test_new_library():
    from mlflow_extensions.serving.engines.vllm_engine import VLLMEngineConfig

    cfg = VLLMEngineConfig(model="foo", library_overrides={"torch": "torch==1.9.0"})
    pip_reqs = cfg.default_pip_reqs()
    assert "torch==1.9.0" in pip_reqs, "torch override not found"


@pytest.mark.unit
def test_override_library():
    from mlflow_extensions.serving.engines.vllm_engine import VLLMEngineConfig

    cfg = VLLMEngineConfig(model="foo", library_overrides={"vllm": "vllm=fooversion"})
    pip_reqs = cfg.default_pip_reqs()
    assert "vllm=fooversion" in pip_reqs, "torch override not found"
