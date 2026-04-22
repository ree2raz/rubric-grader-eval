"""
LLM provider abstraction.

Supports Anthropic (default), OpenAI, and local vLLM endpoints via a
factory function. Each provider implements the same interface: take a
system prompt and user message, return a string response.

No streaming. No retries. No caching. This is a reference implementation.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv

load_dotenv()


class LLMClient(ABC):
    """Base interface for LLM providers."""

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Send a system+user prompt pair and return the text response."""
        ...


class AnthropicClient(LLMClient):
    """Anthropic Claude via the official SDK."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


class OpenAIClient(LLMClient):
    """OpenAI API via the official SDK."""

    def __init__(self, model: str = "gpt-4o"):
        import openai

        self.client = openai.OpenAI()
        self.model = model

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content


class VLLMClient(LLMClient):
    """Local vLLM endpoint, accessed via OpenAI-compatible API.

    Set VLLM_BASE_URL (e.g., http://localhost:8000/v1) and optionally
    VLLM_MODEL to configure. This is the pattern the blog post describes
    for on-prem batch evaluation.
    """

    def __init__(self):
        import openai

        base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.client = openai.OpenAI(base_url=base_url, api_key="not-needed")
        self.model = os.environ.get("VLLM_MODEL", "default")

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content


# Provider registry. Add new providers here.
_PROVIDERS: dict[str, type[LLMClient]] = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "vllm": VLLMClient,
}


def get_llm_client(provider: str | None = None) -> LLMClient:
    """Factory: return an LLM client for the given provider name.

    Resolution order:
    1. Explicit `provider` argument
    2. LLM_PROVIDER environment variable
    3. Default to "anthropic"

    Raises ValueError if the provider name is not recognized.
    """
    name = provider or os.environ.get("LLM_PROVIDER", "anthropic")
    name = name.lower().strip()

    if name not in _PROVIDERS:
        valid = ", ".join(sorted(_PROVIDERS.keys()))
        raise ValueError(
            f"Unknown LLM provider '{name}'. Valid providers: {valid}"
        )

    return _PROVIDERS[name]()
