"""Shared OpenAI Agents SDK model construction."""

from __future__ import annotations

from typing import cast

from agents import Model, ModelSettings, OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI
from openai.types.shared import Reasoning, ReasoningEffort

from .config import AgentConfig

_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)


def build_agent_model(
    config: AgentConfig,
    *,
    json_mode: bool = False,
) -> tuple[Model, ModelSettings, AsyncOpenAI]:
    """Build the configured model and return its explicitly owned client."""

    provider = config.llm_provider.lower()
    if provider == "openai_agents":
        if not config.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai_agents"
            )
        if not config.openai_base_url or not config.openai_model:
            raise ValueError("OPENAI_BASE_URL and OPENAI_MODEL must not be empty")
        if config.reasoning_effort not in _REASONING_EFFORTS:
            raise ValueError(
                "OPENAI_REASONING_EFFORT must be one of: "
                + ", ".join(sorted(_REASONING_EFFORTS))
            )
        client = AsyncOpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
        )
        responses_model = OpenAIResponsesModel(
            model=config.openai_model,
            openai_client=client,
        )
        effort = cast(ReasoningEffort, config.reasoning_effort)
        settings = ModelSettings(
            reasoning=Reasoning(effort=effort),
            verbosity="low",
        )
        return responses_model, settings, client

    if provider == "deepseek":
        if not config.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek")
        if not config.deepseek_base_url or not config.deepseek_model:
            raise ValueError("DEEPSEEK_BASE_URL and DEEPSEEK_MODEL must not be empty")
        client = AsyncOpenAI(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
        )
        chat_model = OpenAIChatCompletionsModel(
            model=config.deepseek_model,
            openai_client=client,
        )
        extra_args = {"response_format": {"type": "json_object"}} if json_mode else None
        return chat_model, ModelSettings(temperature=0.1, extra_args=extra_args), client

    raise ValueError("Unsupported LLM_PROVIDER. Use 'openai_agents' or 'deepseek'.")
