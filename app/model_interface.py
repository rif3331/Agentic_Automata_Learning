"""
General model interface for Agentic Automata Induction.

A model does NOT need to use LangChain.
A model does NOT need to know anything about DFA learning.
A model only needs to implement this runtime contract:

    send(text: str, step: int | None = None) -> dict

The returned dict must contain at least:

    {
        "content": "the exact text produced by the model"
    }

The content is the text that the game parser reads. If the model wants to call a tool,
it must put the tool call in this text, for example:

    <TOOL_ACTION>
    {"tool_name": "is_word_in_language", "input": {"word": "abba"}}
    </TOOL_ACTION>

Optional returned fields:

    raw:      provider-specific raw response object
    thoughts: provider-specific hidden/visible reasoning text, if available
    usage:    provider-specific token usage metadata, if available

Any new provider can run in the game if it is wrapped by an adapter implementing
this interface.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol, Tuple, runtime_checkable

JsonDict = Dict[str, Any]


@runtime_checkable
class ChatModelInterface(Protocol):
    """
    Minimal interface required by llm_runtime.py.

    Required method:
        send(text, step=None) -> dict with at least {"content": str}
    """

    model_name: str

    def send(self, text: str, step: Optional[int] = None) -> JsonDict:
        ...


class GenericTextModelAdapter:
    """
    Generic adapter for any API/function that receives a prompt and returns text.

    generate_fn signature:
        generate_fn(prompt: str, step: int | None = None) -> str

    or:
        generate_fn(prompt: str, step: int | None = None) -> (text, raw_response)

    Example:
        def call_my_model(prompt, step=None):
            raw = my_client.generate(prompt)
            return raw.text, raw

        model = GenericTextModelAdapter(
            model_name="my-model",
            generate_fn=call_my_model,
        )
    """

    def __init__(self, model_name: str, generate_fn: Callable[..., Any]):
        self.model_name = model_name
        self.generate_fn = generate_fn

    def send(self, text: str, step: Optional[int] = None) -> JsonDict:
        result = self.generate_fn(text, step=step)

        raw = result
        thoughts = ""

        if isinstance(result, tuple):
            if len(result) >= 1:
                content = result[0]
            else:
                content = ""
            if len(result) >= 2:
                raw = result[1]
            if len(result) >= 3:
                thoughts = result[2] or ""
        else:
            content = result

        return {
            "content": "" if content is None else str(content),
            "raw": raw,
            "thoughts": thoughts,
        }


class OpenAICompatibleChatAdapter:
    """
    Adapter for any provider exposing the OpenAI Chat Completions API.

    This covers many providers with only configuration changes:
        - OpenAI
        - OpenRouter
        - Together
        - Groq
        - Fireworks
        - vLLM server
        - local OpenAI-compatible servers
        - many hosted Llama/Qwen/Mistral endpoints

    Required configuration:
        api_key
        model_name
        base_url, unless provider is exactly OpenAI
    """

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        base_url: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ):
        from openai import OpenAI

        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.generation_config = generation_config or {}
        self._messages: list[dict[str, str]] = []

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def send(self, text: str, step: Optional[int] = None) -> JsonDict:
        self._messages.append({"role": "user", "content": text})

        allowed = {
            "temperature",
            "top_p",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "stop",
        }
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": self._messages,
        }
        for key, value in self.generation_config.items():
            if key in allowed and value is not None:
                kwargs[key] = value

        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message if getattr(resp, "choices", None) else None
        content = getattr(msg, "content", "") if msg is not None else ""
        reasoning = getattr(msg, "reasoning_content", "") if msg is not None else ""

        self._messages.append({"role": "assistant", "content": content or ""})

        return {
            "content": content or "",
            "raw": resp,
            "thoughts": reasoning or "",
            "usage": getattr(resp, "usage", None),
        }
