"""
Responsible for:

- Creating chat model adapters from API_PROVIDER + MODEL_NAME + API_KEY
- Keeping a registry of supported providers
- Building LangChain-based and OpenAI-compatible model clients
- Providing a clear extension point for new model providers

New model/provider rule:
    Any adapter can run in the game if it implements:

        send(text: str, step: int | None = None) -> dict

    and returns at least:

        {"content": "model output text"}

    The content string is parsed for <TOOL_ACTION>...</TOOL_ACTION>.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

from model_registry import get_provider_spec, normalize_provider
from utils import parse_model_name_and_inline_config

ApiKeyType = Union[str, Dict[str, str]]


def _get_key(api_key: ApiKeyType, provider: str) -> Optional[str]:
    """
    Resolve the single API key used by the selected provider.

    Final single-key mode:
        - The caller passes exactly one key through API_KEY / --api-key.
        - The same variable is used for Gemini, OpenAI, DeepSeek, OpenRouter, etc.
        - This function does NOT read provider-specific environment variables such as
          GEMINI_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, etc.

    Optional single environment variables:
        - If api_key is empty, API_KEY is used as the first single fallback.
        - MODEL_API_KEY is also supported as an alias.

    Provider-specific key dictionaries are intentionally rejected so the runtime
    cannot silently fall back to old per-provider keys.
    """
    _ = normalize_provider(provider)

    if isinstance(api_key, dict):
        raise ValueError(
            "Provider-specific API-key dictionaries are no longer supported. "
            "Use one key only: API_KEY in constants.py, --api-key on the command line, "
            "or the MODEL_API_KEY environment variable."
        )

    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()

    for env_name in ("API_KEY", "MODEL_API_KEY"):
        v = os.environ.get(env_name, "").strip()
        if v:
            return v

    return None


def _ensure_url_scheme(base_url: str) -> str:
    """Return a URL acceptable to OpenAI-compatible clients.

    API_PROVIDER may be written as either:
        - "http://127.0.0.1:8000/v1"
        - "127.0.0.1:8000/v1"
        - "localhost:8000/v1"

    For convenience, missing schemes are treated as local/plain HTTP endpoints.
    """
    value = (base_url or "").strip()
    if not value:
        return value
    if value.startswith(("http://", "https://")):
        return value
    return f"http://{value}"


def _provider_base_url(provider: str, generation_config: Optional[Dict[str, Any]]) -> Optional[str]:
    gen = generation_config or {}
    base_url = gen.get("base_url") or gen.get("api_base")
    if isinstance(base_url, str) and base_url.strip():
        return _ensure_url_scheme(base_url)

    normalized_provider = normalize_provider(provider)
    if normalized_provider == "unknown_provider":
        raw_provider = str(gen.get("api_provider") or provider or "").strip()
        if raw_provider and raw_provider != "unknown_provider":
            return _ensure_url_scheme(raw_provider)

    return get_provider_spec(provider).default_base_url


def _strip_provider_prefix(model_name: str, provider: str) -> str:
    """Accept old-style MODEL_NAME values like 'openai:gpt-5.4'.

    The new architecture uses API_PROVIDER separately from MODEL_NAME, but this
    keeps old commented examples from breaking the run. If the prefix matches
    the selected provider, it is removed. If the prefix is different, we still
    remove it and trust API_PROVIDER as the source of truth.
    """
    m = (model_name or "").strip()
    if ":" not in m:
        return m
    maybe_provider, maybe_model = m.split(":", 1)
    if maybe_model.strip():
        return maybe_model.strip()
    return m


def _build_langchain_chat_model(
    *,
    provider: str,
    model: str,
    api_key: ApiKeyType,
    generation_config: Optional[Dict[str, Any]],
    safety_settings: Optional[List[Dict[str, Any]]],
):
    provider = normalize_provider(provider)
    spec = get_provider_spec(provider)
    gen = generation_config or {}

    internal_keys = {
        "require_context_cache",
        "context_cache_ttl",
        "cache_padding_target_tokens",
        "min_cache_tokens",
        "initial_cache_padding_target_tokens",
        "cache_padding_step_down",
        "print_cache_events",
        "print_tool_io",
        "cache_update_threshold",
        "first_cache_create_step",
        "api_provider",
        "actual_model",
        "display_model_name",
    }

    common_keys = {
        "temperature", "top_p", "timeout", "max_retries", "stop", "seed",
        "presence_penalty", "frequency_penalty", "top_k", "base_url", "api_base",
        "max_tokens", "reasoning", "output_version",
    }

    direct_kwargs: Dict[str, Any] = {}
    model_kwargs: Dict[str, Any] = {}

    for k, v in gen.items():
        if k in internal_keys:
            continue
        if k == "include_thoughts":
            continue
        if k in common_keys:
            direct_kwargs[k] = v
        else:
            model_kwargs[k] = v

    direct_kwargs.pop("api_base", None)

    if model_kwargs:
        direct_kwargs["model_kwargs"] = model_kwargs

    if spec.kind == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = _get_key(api_key, provider)
        if not key:
            raise ValueError(f"Missing API key for provider '{provider}'.")
        kwargs = {"model": model, "api_key": key, **direct_kwargs}
        if safety_settings is not None:
            kwargs["safety_settings"] = safety_settings
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        key = _get_key(api_key, provider)
        if not key:
            raise ValueError("Missing API key for provider 'openai'.")

        actual_model = gen.get("actual_model", model)
        return ChatOpenAI(model=actual_model, api_key=key, **direct_kwargs)

    if spec.kind == "anthropic":
        key = _get_key(api_key, provider)
        if not key:
            raise ValueError(f"Missing API key for provider '{provider}'.")

        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                "Provider 'anthropic' requires the package 'langchain-anthropic'. "
                "Install it with: pip install langchain-anthropic"
            ) from e

        # ChatAnthropic accepts model/model_name depending on package version.
        # model is supported by current LangChain Anthropic versions.
        return ChatAnthropic(model=model, api_key=key, **direct_kwargs)

    if spec.kind in {"openai_compatible_langchain", "deepseek"}:
        from langchain_openai import ChatOpenAI
        key = _get_key(api_key, provider)
        if not key:
            raise ValueError(f"Missing API key for provider '{provider}'.")

        base_url = _provider_base_url(provider, gen)
        if not base_url:
            raise ValueError(
                f"Provider '{provider}' requires a base URL. "
                "For a local/OpenAI-compatible server, put the server URL directly in API_PROVIDER, "
                "for example API_PROVIDER='127.0.0.1:8000/v1'."
            )

        direct_kwargs.pop("base_url", None)

        return ChatOpenAI(
            model=model,
            api_key=key,
            base_url=base_url,
            **direct_kwargs,
        )

    raise ValueError(f"Provider '{provider}' is registered but not implemented in model_factory.py.")


def build_model(
    *,
    api_key: ApiKeyType,
    model_name: str,
    mode: str = "chat",
    api_provider: Optional[str] = None,
    system_prompt: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    safety_settings: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    print_memory: bool = True,
    print_tool_action_once_per_step: bool = True,
    print_raw_model_output: bool = False,
):
    """
    Main constructor.

    New recommended usage:
        build_model(
            api_provider="openrouter",
            model_name="anthropic/claude-...",
            api_key="...",
        )

    Backward-compatible usage still works:
        build_model(model_name="gemini:gemini-...", api_key={...})
    """
    from model_router import GeminiModelRouter

    mode = "chat"

    gen = dict(generation_config or {})
    if api_provider is not None:
        gen["api_provider"] = api_provider
    if temperature is not None:
        gen["temperature"] = float(temperature)
    if seed is not None:
        gen["seed"] = int(seed)

    if api_provider is not None:
        model_name = _strip_provider_prefix(model_name, api_provider)
    model_name, inline_model_config = parse_model_name_and_inline_config(model_name)
    gen.update(inline_model_config)

    return GeminiModelRouter(
        api_key=api_key,
        api_provider=api_provider,
        model_name=model_name,
        mode=mode,
        system_prompt=system_prompt,
        generation_config=gen,
        safety_settings=safety_settings,
        print_memory=print_memory,
        print_tool_action_once_per_step=print_tool_action_once_per_step,
        print_raw_model_output=print_raw_model_output,
    )


def build_llm_from_provider_model(
    *,
    api_provider: str,
    model_name: str,
    api_key: str,
    mode: str = "chat",
    system_prompt: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    safety_settings: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
):
    """
    Recommended single-key entry point.

    Only three user-level variables are needed:
        API_PROVIDER = "gemini" / "openai" / "openrouter" / a local OpenAI-compatible URL / ...
        MODEL_NAME = "the exact provider model id"
        API_KEY = "the key for that provider"
    """
    return build_model(
        api_provider=api_provider,
        api_key=api_key,
        model_name=model_name,
        mode=mode,
        system_prompt=system_prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        temperature=temperature,
        seed=seed,
        print_tool_action_once_per_step=False,
        print_raw_model_output=True,
    )


def build_llm_from_model_name(
    *,
    model_name: str,
    mode: str = "chat",
    system_prompt: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    safety_settings: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    api_provider: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Single-key compatibility builder.

    Use only:
        api_provider = API_PROVIDER
        model_name = MODEL_NAME
        api_key = API_KEY

    Provider-specific keys such as api_key_gemini/api_key_openai are no longer
    part of the supported path.
    """
    if api_provider is None:
        provider, real_model = __import__("utils").parse_model_name(model_name)
        api_provider = provider
        model_name = real_model

    return build_llm_from_provider_model(
        api_provider=api_provider,
        model_name=model_name,
        api_key=api_key or "",
        mode=mode,
        system_prompt=system_prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        temperature=temperature,
        seed=seed,
    )
