import pytest

from groundwork.compliance import ComplianceError, check_data_source, find_restriction
from groundwork.config import EndpointConfig, load_config


def test_example_config_loads(root):
    cfg = load_config(root / "configs/example.yaml", ["train.lr=2e-5", "agent.max_steps=3", "runtime.guard.enabled=false"])
    assert cfg.train.lr == 2e-5 and isinstance(cfg.train.lr, float)
    assert cfg.agent.max_steps == 3
    assert cfg.runtime.guard.enabled is False
    assert cfg.teacher is None


def test_unknown_key_is_an_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  learning_rate: 1e-5\n")
    with pytest.raises(ValueError, match="unknown key"):
        load_config(p)


@pytest.mark.parametrize("base_url,model", [
    ("https://api.anthropic.com/v1", "anything"),
    ("https://api.openai.com/v1", "anything"),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "anything"),
    ("https://my-gateway.example.com/v1", "anthropic/claude-x"),
    ("https://my-gateway.example.com/v1", "gpt-4o"),
    ("https://my-gateway.example.com/v1", "openai/gpt-5"),
    ("https://my-gateway.example.com/v1", "o3-mini"),
    ("https://my-gateway.example.com/v1", "google/gemini-2.5-pro"),
    ("https://bedrock.example.com/v1", "anthropic.claude-v2"),
])
def test_restricted_sources_are_refused(base_url, model):
    ep = EndpointConfig(base_url=base_url, model=model, ownership="self")
    assert find_restriction(ep) is not None
    with pytest.raises(ComplianceError):
        check_data_source(ep, "teacher")


@pytest.mark.parametrize("model", ["openai/gpt-oss-120b", "gpt-oss-20b", "google/gemma-3-27b-it", "Qwen/Qwen3-32B", "deepseek-reasoner", "my-model"])
def test_open_and_own_models_are_allowed(model):
    ep = EndpointConfig(base_url="http://localhost:8000/v1", model=model, ownership="self")
    assert find_restriction(ep) is None
    check_data_source(ep, "model")


def test_third_party_needs_explicit_terms():
    ep = EndpointConfig(base_url="https://api.example.com/v1", model="some-model", ownership="third_party")
    with pytest.raises(ComplianceError, match="distillation_permitted"):
        check_data_source(ep, "teacher")
    ep.provider, ep.terms_url, ep.distillation_permitted = "Example", "https://example.com/terms", True
    check_data_source(ep, "teacher")
