"""
Provider registry for model creation.

Main idea:
    API_PROVIDER chooses the API/provider implementation.
    MODEL_NAME chooses the concrete model inside that provider.
    API_KEY is the single key passed to the selected provider. No provider-specific API-key variables are used.

For a new provider, add a registry entry or create an adapter that implements:

    send(text: str, step: int | None = None) -> dict

and returns:

    {"content": "model output text"}

The model output text must contain <TOOL_ACTION>...</TOOL_ACTION> when the model
wants to call a tool.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    kind: str
    default_base_url: Optional[str] = None
    api_key_env: Optional[str] = None  # deprecated; single key is API_KEY / MODEL_API_KEY
    description: str = ""


PROVIDER_REGISTRY: Dict[str, ProviderSpec] = {
    "gemini": ProviderSpec(
        name="gemini",
        kind="gemini",
        description="Google Gemini / Google GenAI provider.",
    ),
    "google": ProviderSpec(
        name="google",
        kind="gemini",
        description="Alias for Gemini.",
    ),
    "openai": ProviderSpec(
        name="openai",
        kind="langchain_openai",
        description="OpenAI provider.",
    ),
    "deepseek": ProviderSpec(
        name="deepseek",
        kind="deepseek",
        default_base_url="https://api.deepseek.com",
        description="DeepSeek provider using OpenAI-compatible API.",
    ),
    "together": ProviderSpec(
        name="together",
        kind="openai_compatible_langchain",
        default_base_url="https://api.together.xyz/v1",
        description="Together provider using OpenAI-compatible API.",
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        kind="openai_compatible_langchain",
        default_base_url="https://openrouter.ai/api/v1",
        description="OpenRouter provider using OpenAI-compatible API.",
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        kind="anthropic",
        description="Anthropic Claude provider. Uses the Anthropic chat adapter.",
    ),
    "claude": ProviderSpec(
        name="claude",
        kind="anthropic",
        description="Alias for Anthropic Claude provider.",
    ),
    "grok": ProviderSpec(
        name="grok",
        kind="openai_compatible_langchain",
        default_base_url="https://api.x.ai/v1",
        description="xAI Grok provider using OpenAI-compatible API.",
    ),
    "xai": ProviderSpec(
        name="xai",
        kind="openai_compatible_langchain",
        default_base_url="https://api.x.ai/v1",
        description="xAI provider using OpenAI-compatible API.",
    ),
    "groq": ProviderSpec(
        name="groq",
        kind="openai_compatible_langchain",
        default_base_url="https://api.groq.com/openai/v1",
        description="Groq provider using OpenAI-compatible API.",
    ),
    "fireworks": ProviderSpec(
        name="fireworks",
        kind="openai_compatible_langchain",
        default_base_url="https://api.fireworks.ai/inference/v1",
        description="Fireworks provider using OpenAI-compatible API.",
    ),
    "unknown_provider": ProviderSpec(
        name="unknown_provider",
        kind="openai_compatible_langchain",
        default_base_url=None,
        description=(
            "Fallback for an unregistered API_PROVIDER. The raw API_PROVIDER string "
            "is treated as an OpenAI-compatible base URL. If it does not include "
            "http:// or https://, http:// is added automatically."
        ),
    ),
}


def normalize_provider(provider: str | None) -> str:
    p = (provider or "gemini").strip().lower()
    aliases = {
        "google_genai": "gemini",
        "google-ai": "gemini",
        "google_ai": "gemini",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "x-ai": "xai",
        "x.ai": "xai",
        "xai": "xai",
        "grok": "grok",
    }
    p = aliases.get(p, p)
    if p not in PROVIDER_REGISTRY:
        # Any unregistered provider value is interpreted as an OpenAI-compatible
        # server URL supplied directly through API_PROVIDER. This intentionally
        # does not require the value to start with http:// or https://.
        return "unknown_provider"
    return p


def get_provider_spec(provider: str | None) -> ProviderSpec:
    return PROVIDER_REGISTRY[normalize_provider(provider)]


def explain_model_interface() -> str:
    return (
        "Every model adapter must implement: "
        "send(text: str, step: int | None = None) -> dict. "
        "The returned dict must contain {'content': <model output text>}. "
        "The model output text is parsed for <TOOL_ACTION>...</TOOL_ACTION>."
    )
