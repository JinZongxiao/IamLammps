"""Environment configuration contracts."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lammps_agent.config import AgentConfig, load_config
from lammps_agent.model_provider import build_agent_model


def _model_config(provider: str) -> AgentConfig:
    return AgentConfig(
        llm_provider=provider,
        openai_base_url="https://api.openai.test/v1",
        openai_api_key="openai-key",
        openai_model="gpt-test",
        lammps_command="lmp",
        enabled_analyzers=[],
        deepseek_base_url="https://api.deepseek.test/v1",
        deepseek_api_key="deepseek-key",
        deepseek_model="deepseek-test",
    )


def test_loads_deepseek_provider_settings(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.deepseek/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")

    config = load_config()

    assert config.llm_provider == "deepseek"
    assert config.deepseek_base_url == "https://example.deepseek/v1"
    assert config.deepseek_api_key == "test-deepseek-key"
    assert config.deepseek_model == "deepseek-chat"


def test_deepseek_defaults_to_v4_flash(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    assert load_config().deepseek_model == "deepseek-v4-flash"


def test_openai_requires_explicit_key_and_model(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "openai_agents")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    config = load_config()

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_agent_model(config)

    config.openai_api_key = "test-key"
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        build_agent_model(config)


def test_builds_openai_responses_model_with_owned_client() -> None:
    config = _model_config("openai_agents")
    client = object()
    model = object()

    with (
        patch("lammps_agent.model_provider.AsyncOpenAI", return_value=client) as client_type,
        patch(
            "lammps_agent.model_provider.OpenAIResponsesModel", return_value=model
        ) as model_type,
    ):
        built_model, settings, built_client = build_agent_model(config)

    assert built_model is model
    assert built_client is client
    assert settings.reasoning is not None
    assert settings.reasoning.effort == "medium"
    assert settings.verbosity == "low"
    client_type.assert_called_once_with(
        api_key="openai-key", base_url="https://api.openai.test/v1"
    )
    model_type.assert_called_once_with(model="gpt-test", openai_client=client)


def test_rejects_invalid_reasoning_before_allocating_client() -> None:
    config = _model_config("openai_agents")
    config.reasoning_effort = "extreme"

    with patch("lammps_agent.model_provider.AsyncOpenAI") as client_type:
        with pytest.raises(ValueError, match="OPENAI_REASONING_EFFORT"):
            build_agent_model(config)

    client_type.assert_not_called()


def test_builds_deepseek_chat_model_and_optional_json_mode() -> None:
    config = _model_config("deepseek")
    client = object()
    model = object()

    with (
        patch("lammps_agent.model_provider.AsyncOpenAI", return_value=client),
        patch(
            "lammps_agent.model_provider.OpenAIChatCompletionsModel",
            return_value=model,
        ),
    ):
        built_model, settings, built_client = build_agent_model(config, json_mode=True)

    assert (built_model, built_client) == (model, client)
    assert settings.temperature == 0.1
    assert settings.extra_args == {"response_format": {"type": "json_object"}}


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("deepseek_api_key", "DEEPSEEK_API_KEY"),
        ("deepseek_base_url", "DEEPSEEK_BASE_URL"),
        ("deepseek_model", "DEEPSEEK_BASE_URL"),
    ],
)
def test_deepseek_requires_complete_configuration(field: str, message: str) -> None:
    config = _model_config("deepseek")
    setattr(config, field, "")

    with pytest.raises(ValueError, match=message):
        build_agent_model(config)


def test_rejects_unknown_model_provider() -> None:
    config = _model_config("other")

    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        build_agent_model(config)
