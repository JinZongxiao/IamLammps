from __future__ import annotations

import json
from dataclasses import dataclass

import requests
from openai import OpenAI

from .config import AgentConfig


@dataclass
class ChatMessage:
    role: str
    content: str


class BaseChatClient:
    def chat(self, messages: list[ChatMessage], temperature: float = 0.1) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(BaseChatClient):
    def __init__(self, config: AgentConfig) -> None:
        self.model = config.openai_model
        self.base_url = config.openai_base_url.rstrip("/")
        self.api_key = config.openai_api_key
        # Many local OpenAI-compatible servers do not enforce API keys.
        sdk_api_key = config.openai_api_key or "EMPTY"
        self.client = OpenAI(base_url=config.openai_base_url, api_key=sdk_api_key)

    def chat(self, messages: list[ChatMessage], temperature: float = 0.1) -> str:
        payload_messages = [{"role": m.role, "content": m.content} for m in messages]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=payload_messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            if isinstance(content, str):
                return content
            if content is None:
                return ""
            return str(content)
        except Exception:
            # Some "OpenAI-compatible" gateways are only partially compatible with SDK behavior.
            url = f"{self.base_url}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            resp = requests.post(
                url,
                headers=headers,
                json={
                    "model": self.model,
                    "messages": payload_messages,
                    "temperature": temperature,
                },
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()

            choices = data.get("choices") or []
            if not choices:
                return json.dumps(data, ensure_ascii=True)

            message = choices[0].get("message") or {}
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") in {"text", None}
                ]
                return "".join(text_parts)

            return json.dumps(data, ensure_ascii=True)


class LocalOllamaClient(BaseChatClient):
    def __init__(self, config: AgentConfig) -> None:
        self.base_url = config.local_base_url.rstrip("/")
        self.model = config.local_model

    def chat(self, messages: list[ChatMessage], temperature: float = 0.1) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }

        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()

        message = data.get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content

        # Fallback for unexpected payload shapes.
        return json.dumps(data, ensure_ascii=True)


def build_chat_client(config: AgentConfig) -> BaseChatClient:
    provider = config.llm_provider.lower()
    if provider == "openai_compatible":
        return OpenAICompatibleClient(config)
    if provider == "local_ollama":
        return LocalOllamaClient(config)

    raise ValueError(
        "Unsupported LLM_PROVIDER. Use 'openai_compatible' or 'local_ollama'."
    )
