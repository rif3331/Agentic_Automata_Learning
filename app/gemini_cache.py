"""
Responsible for:

- Managing Gemini-specific context caching and cache lifecycle operations
- Building Gemini generation configurations and provider request parameters
- Formatting transcripts and padding content for context-cache token requirements
- Creating, updating, deleting, and tracking Gemini cached content usage
- Handling Gemini generation flows with and without context caching
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

def build_gemini_generate_config(owner: Any, *, cached_content: Optional[str] = None):
    kwargs: Dict[str, Any] = {}

    temp = owner._generation_config.get("temperature", None)
    if temp is not None:
        kwargs["temperature"] = temp

    top_p = owner._generation_config.get("top_p", None)
    if top_p is not None:
        kwargs["top_p"] = top_p

    top_k = owner._generation_config.get("top_k", None)
    if top_k is not None:
        kwargs["top_k"] = top_k

    seed = owner._generation_config.get("seed", None)
    if seed is not None:
        kwargs["seed"] = seed

    thinking_cfg = {}

    if owner._include_thoughts:
        thinking_cfg["include_thoughts"] = True

    thinking_level = owner._generation_config.get("thinking_level", None)
    if thinking_level is not None:
        thinking_cfg["thinking_level"] = thinking_level

    thinking_budget = owner._generation_config.get("thinking_budget", None)
    if thinking_budget is not None:
        thinking_cfg["thinking_budget"] = thinking_budget

    if thinking_cfg:
        kwargs["thinking_config"] = thinking_cfg

    if cached_content:
        kwargs["cached_content"] = cached_content

    return owner._gemini_types.GenerateContentConfig(**kwargs)


def normalize_gemini_model_id(owner: Any, model: str) -> str:
    m = (model or "").strip()
    return m if m.startswith("models/") else f"models/{m}"


def format_transcript_for_cache(owner: Any) -> str:
    parts = []
    if owner.system_prompt:
        parts.append("SYSTEM:\n" + owner.system_prompt)

    msgs_str = owner._messages_to_string(owner._messages)
    if msgs_str:
        parts.append(msgs_str)

    return "\n\n".join(parts).strip()


def pad_to_min_tokens(owner: Any, s: str) -> str:
    # No artificial cache padding. Keep the transcript exactly as-is.
    return s or ""


def delete_cache(owner: Any) -> bool:
    name = owner._cached_content_name
    owner._cached_message_count = 0
    if not (owner._require_context_cache and owner._provider in ("gemini", "google")):
        return False
    if not name or not owner._gemini_client:
        owner._cached_content_name = None
        return False
    try:
        owner._gemini_client.caches.delete(name)
        return True
    except Exception:
        return False
    finally:
        owner._cached_content_name = None


def create_or_replace_cache(owner: Any) -> None:
    if not owner._require_context_cache:
        return

    transcript_raw = owner._format_transcript_for_cache()
    transcript = owner._pad_to_min_tokens(transcript_raw)

    if owner._cached_content_name:
        try:
            owner._gemini_client.caches.delete(owner._cached_content_name)
        except Exception:
            pass

    cache = owner._gemini_client.caches.create(
        model=owner._normalize_gemini_model_id(owner._real_model),
        config=owner._gemini_types.CreateCachedContentConfig(
            display_name="chat_game_cache",
            contents=[transcript],
            ttl=str(owner._cache_ttl),
        ),
    )

    owner._cached_content_name = cache.name
    owner._cached_message_count = len(owner._messages)

    retrieved = owner._gemini_client.caches.get(name=owner._cached_content_name)
    um = getattr(retrieved, "usage_metadata", None)
    token_count = getattr(um, "total_token_count", None) if um is not None else None
    if not isinstance(token_count, int):
        raise ValueError("Failed to read total_token_count from caches.get().")

    owner.cache_token_counts.append(token_count)

    if isinstance(owner.cache_stats, dict):
        owner.cache_stats["cache_creates"] = int(owner.cache_stats.get("cache_creates", 0) or 0) + 1
        owner.cache_stats["cache_create_tokens_total"] = int(owner.cache_stats.get("cache_create_tokens_total", 0) or 0) + token_count

    owner.cache_data = {
        "cache_count": len(owner.cache_token_counts),
        "total_tokens": int(sum(owner.cache_token_counts)),
        "token_counts": list(owner.cache_token_counts),
    }
    owner.__dict__["cache_data"] = owner.cache_data

    owner._cache_debug[owner._cached_content_name] = {
        "transcript_raw": transcript_raw,
        "target": owner._current_cache_padding_target_tokens,
    }

    owner._current_cache_padding_target_tokens = max(
        0, owner._current_cache_padding_target_tokens - owner._cache_padding_step_down
    )


def gemini_generate_with_cache(owner: Any, user_text: str) -> Tuple[str, str, Any]:
    cfg = owner._build_gemini_generate_config(
        cached_content=owner._cached_content_name if owner._cached_content_name else None
    )

    resp = owner._gemini_client.models.generate_content(
        model=owner._normalize_gemini_model_id(owner._real_model),
        contents=user_text,
        config=cfg,
    )

    answer_text, thoughts_text = owner._extract_gemini_text_and_thoughts(resp)
    return answer_text, thoughts_text, resp


def gemini_generate_no_cache(owner: Any, user_text: str) -> Tuple[str, str, Any]:
    cfg = owner._build_gemini_generate_config()

    resp = owner._gemini_client.models.generate_content(
        model=owner._normalize_gemini_model_id(owner._real_model),
        contents=user_text,
        config=cfg,
    )

    answer_text, thoughts_text = owner._extract_gemini_text_and_thoughts(resp)
    return answer_text, thoughts_text, resp