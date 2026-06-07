"""
Responsible for:

- Managing model routing across Gemini, DeepSeek, and other chat providers
- Handling model requests, responses, tool-call extraction, and conversation memory
- Managing Gemini context caching, cache lifecycle, and transcript formatting
- Tracking response latency, token usage, reasoning tokens, and per-step metrics
- Coordinating provider-specific generation flows and normalizing model outputs
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import time
import atexit
from utils import parse_model_name
from utils import parse_model_name_and_inline_config
from model_registry import normalize_provider, get_provider_spec
from model_factory import ApiKeyType, _get_key, _build_langchain_chat_model
from gemini_cache import (
    build_gemini_generate_config as build_gemini_generate_config_impl,
    normalize_gemini_model_id as normalize_gemini_model_id_impl,
    format_transcript_for_cache as format_transcript_for_cache_impl,
    pad_to_min_tokens as pad_to_min_tokens_impl,
    delete_cache as delete_cache_impl,
    create_or_replace_cache as create_or_replace_cache_impl,
    gemini_generate_with_cache as gemini_generate_with_cache_impl,
    gemini_generate_no_cache as gemini_generate_no_cache_impl,
)
from response_parsing import (
    extract_openai_text_and_thoughts as extract_openai_text_and_thoughts_impl,
    extract_gemini_text_and_thoughts as extract_gemini_text_and_thoughts_impl,
    coerce_text as coerce_text_impl,
    print_raw_model_text as print_raw_model_text_impl,
    print_model_raw as print_model_raw_impl,
    repair_tool_json as repair_tool_json_impl,
    safe_json_loads as safe_json_loads_impl,
    strip_knowledge_state_from_tool_result as strip_knowledge_state_from_tool_result_impl,
    messages_to_string as messages_to_string_impl,
    messages_to_openai_format as messages_to_openai_format_impl,
    from_text as from_text_impl,
    extract_tool_actions_from_text as extract_tool_actions_from_text_impl,
)

JsonDict = Dict[str, Any]


class GeminiModelRouter:
    def __init__(
        self,
        api_key: ApiKeyType,
        model_name: str,
        mode: str = "chat",
        api_provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        print_memory: bool = True,
        print_tool_action_once_per_step: bool = True,
        print_raw_model_output: bool = False,
    ):
        from langchain_core.messages import AIMessage, HumanMessage

        self._AIMessage = AIMessage
        self._HumanMessage = HumanMessage

        self.mode = "chat"
        self.system_prompt = system_prompt
        self._generation_config = generation_config or {}
        self.model_name = str(
            self._generation_config.get("display_model_name")
            or model_name
            or ""
        )

        # New recommended mode: API_PROVIDER is separate from MODEL_NAME.
        # Backward-compatible mode: if api_provider is omitted, model_name may still
        # be written as "provider:model-id" and will be parsed here.
        if api_provider is not None:
            provider = normalize_provider(api_provider)
            real_model = (model_name or "").strip()
            # Backward compatibility: old configs sometimes wrote MODEL_NAME as
            # "provider:model-id". In the new architecture API_PROVIDER already
            # carries the provider, so strip any prefix before passing the model
            # id to the provider SDK.
            if ":" in real_model:
                _old_provider, _old_model = real_model.split(":", 1)
                if _old_model.strip():
                    real_model = _old_model.strip()
            real_model, inline_model_config = parse_model_name_and_inline_config(real_model)
            self._generation_config.update(inline_model_config)
        else:
            provider, real_model = parse_model_name(model_name)
            provider = normalize_provider(provider)
            real_model, inline_model_config = parse_model_name_and_inline_config(real_model)
            self._generation_config.update(inline_model_config)

        self.api_provider = provider
        self._provider = provider
        self._provider_spec = get_provider_spec(provider)
        self._real_model = real_model

        self._include_thoughts = bool(self._generation_config.get("include_thoughts", False))
        self.step_thought_texts: Dict[int, str] = {}
        self._require_context_cache = bool(self._generation_config.get("require_context_cache", False))
        self._cache_ttl = self._generation_config.get("context_cache_ttl", "1800s")

        self._initial_cache_padding_target_tokens = int(self._generation_config.get("initial_cache_padding_target_tokens", -0))
        self._cache_padding_step_down = int(self._generation_config.get("cache_padding_step_down", 0))
        self._current_cache_padding_target_tokens = self._initial_cache_padding_target_tokens

        self._min_cache_tokens = int(self._generation_config.get("min_cache_tokens", 0))
        self._cache_update_threshold = int(self._generation_config.get("cache_update_threshold", 10))
        # Do not create the first Gemini context cache before this model step.
        # The transcript is still accumulated in self._messages before this point.
        self._first_cache_create_step = int(self._generation_config.get("first_cache_create_step", 10))

        self._print_cache_events = bool(self._generation_config.get("print_cache_events", False))
        self._print_tool_io = bool(self._generation_config.get("print_tool_io", False))
        self._print_raw_model_output = bool(print_raw_model_output)

        if self._require_context_cache and self._provider_spec.kind != "gemini":
            raise ValueError("Context cache is only supported for the Gemini/Google provider.")

        self._model = _build_langchain_chat_model(
            provider=provider,
            model=real_model,
            api_key=api_key,
            generation_config=self._generation_config,
            safety_settings=safety_settings,
        )

        self._messages: List[Any] = []
        self._started = False
        self._game_prompt_sent = False

        self._print_tool_action_once_per_step = bool(print_tool_action_once_per_step)
        self._last_send_was_tool_result = False

        self._cached_content_name: Optional[str] = None
        self._cached_message_count: int = 0
        self._cache_debug: Dict[str, Dict[str, Any]] = {}

        self._gemini_api_key: Optional[str] = _get_key(api_key, self._provider) if self._provider_spec.kind == "gemini" else None
        self._gemini_client = None
        self._gemini_types = None

        self._deepseek_api_key: Optional[str] = _get_key(api_key, self._provider) if self._provider == "deepseek" else None
        self._deepseek_client = None

        self.cache_stats: Dict[str, int] = {"cache_creates": 0, "cache_create_tokens_total": 0}
        self.cache_token_counts: List[int] = []
        self.cache_data: Dict[str, Any] = {"cache_count": 0, "total_tokens": 0, "token_counts": []}
        self.__dict__["cache_data"] = self.cache_data

        if self._provider_spec.kind == "gemini":
            from google import genai
            from google.genai import types
            self._gemini_client = genai.Client(api_key=self._gemini_api_key)
            self._gemini_types = types
            atexit.register(self.delete_cache)

        if self._provider == "deepseek":
            from openai import OpenAI
            self._deepseek_client = OpenAI(
                api_key=self._deepseek_api_key,
                base_url="https://api.deepseek.com",
            )

    def _extract_openai_text_and_thoughts(self, resp: Any) -> Tuple[str, str]:
        return extract_openai_text_and_thoughts_impl(self, resp)

    def _build_gemini_generate_config(self, *, cached_content: Optional[str] = None):
        return build_gemini_generate_config_impl(self, cached_content=cached_content)

    def _extract_gemini_text_and_thoughts(self, resp: Any) -> Tuple[str, str]:
        return extract_gemini_text_and_thoughts_impl(self, resp)

    def _coerce_text(self, text) -> str:
        return coerce_text_impl(self, text)

    def _print_raw_model_text(self, text: str, *, tag: str = "", step: int | None = None) -> None:
        print_raw_model_text_impl(self, text, tag=tag, step=step)

    def print_model_raw(self, resp: Any) -> None:
        print_model_raw_impl(self, resp)

    def _should_print_cache_event(self, event: str) -> bool:
        return self._print_cache_events

    def start(self):
        if not self._started:
            self._started = True

    def _normalize_gemini_model_id(self, model: str) -> str:
        return normalize_gemini_model_id_impl(self, model)

    def _repair_tool_json(self, s: str) -> str:
        return repair_tool_json_impl(self, s)

    def _safe_json_loads(self, s: str) -> Optional[JsonDict]:
        return safe_json_loads_impl(self, s)

    def _strip_knowledge_state_from_tool_result(self, text: str) -> str:
        return strip_knowledge_state_from_tool_result_impl(self, text)

    def _messages_to_string(self, messages: List[Any]) -> str:
        return messages_to_string_impl(self, messages)

    def _messages_to_openai_format(self, messages: List[Any]) -> List[Dict[str, str]]:
        return messages_to_openai_format_impl(self, messages)

    def _format_transcript_for_cache(self) -> str:
        return format_transcript_for_cache_impl(self)

    def _pad_to_min_tokens(self, s: str) -> str:
        return pad_to_min_tokens_impl(self, s)

    def delete_cache(self) -> bool:
        return delete_cache_impl(self)

    def _create_or_replace_cache(self) -> None:
        create_or_replace_cache_impl(self)

    def _extract_usage_fields(self, resp: Any) -> Dict[str, Any]:
        usage: Dict[str, Any] = {}

        rm = getattr(resp, "response_metadata", None)
        if isinstance(rm, dict):
            tu = rm.get("token_usage")
            if isinstance(tu, dict):
                usage["token_usage"] = tu

        um = getattr(resp, "usage_metadata", None)
        if isinstance(um, dict):
            usage["usage_metadata"] = um

        return usage

    def _usage_to_metrics(self, usage_obj: Any) -> Dict[str, Optional[int]]:
        if usage_obj is not None and not isinstance(usage_obj, dict):
            try:
                usage_obj = usage_obj.model_dump()
            except Exception:
                try:
                    usage_obj = dict(usage_obj)
                except Exception:
                    try:
                        usage_obj = usage_obj.__dict__
                    except Exception:
                        usage_obj = {}

        total_tokens: Optional[int] = None
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None

        thoughts_tokens: Optional[int] = None
        output_visible_tokens: Optional[int] = None
        cache_history_tokens: Optional[int] = None

        if isinstance(usage_obj, dict):
            if isinstance(usage_obj.get("total_token_count"), int):
                total_tokens = usage_obj.get("total_token_count")
            if isinstance(usage_obj.get("prompt_token_count"), int):
                input_tokens = usage_obj.get("prompt_token_count")
            if isinstance(usage_obj.get("candidates_token_count"), int):
                output_visible_tokens = usage_obj.get("candidates_token_count")
            if isinstance(usage_obj.get("thoughts_token_count"), int):
                thoughts_tokens = usage_obj.get("thoughts_token_count")
            if isinstance(usage_obj.get("cached_content_token_count"), int):
                cache_history_tokens = usage_obj.get("cached_content_token_count")

            if isinstance(usage_obj.get("total_tokens"), int) and total_tokens is None:
                total_tokens = usage_obj.get("total_tokens")
            if isinstance(usage_obj.get("prompt_tokens"), int) and input_tokens is None:
                input_tokens = usage_obj.get("prompt_tokens")
            if isinstance(usage_obj.get("completion_tokens"), int) and output_visible_tokens is None:
                output_visible_tokens = usage_obj.get("completion_tokens")

            details0 = usage_obj.get("completion_tokens_details")
            if isinstance(details0, dict) and isinstance(details0.get("reasoning_tokens"), int) and thoughts_tokens is None:
                thoughts_tokens = details0.get("reasoning_tokens")

            ptd0 = usage_obj.get("prompt_tokens_details")
            if isinstance(ptd0, dict) and isinstance(ptd0.get("cached_tokens"), int) and cache_history_tokens is None:
                cache_history_tokens = ptd0.get("cached_tokens")

            if isinstance(usage_obj.get("prompt_cache_hit_tokens"), int) and cache_history_tokens is None:
                cache_history_tokens = usage_obj.get("prompt_cache_hit_tokens")

            tu = usage_obj.get("token_usage")
            if isinstance(tu, dict):
                if isinstance(tu.get("total_tokens"), int) and total_tokens is None:
                    total_tokens = tu.get("total_tokens")
                if isinstance(tu.get("prompt_tokens"), int) and input_tokens is None:
                    input_tokens = tu.get("prompt_tokens")
                if isinstance(tu.get("completion_tokens"), int) and output_visible_tokens is None:
                    output_visible_tokens = tu.get("completion_tokens")

                details = tu.get("completion_tokens_details")
                if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), int) and thoughts_tokens is None:
                    thoughts_tokens = details.get("reasoning_tokens")

                ptd = tu.get("prompt_tokens_details")
                if isinstance(ptd, dict) and isinstance(ptd.get("cached_tokens"), int) and cache_history_tokens is None:
                    cache_history_tokens = ptd.get("cached_tokens")

                if isinstance(tu.get("prompt_cache_hit_tokens"), int) and cache_history_tokens is None:
                    cache_history_tokens = tu.get("prompt_cache_hit_tokens")

            um = usage_obj.get("usage_metadata")
            if isinstance(um, dict):
                if isinstance(um.get("total_tokens"), int) and total_tokens is None:
                    total_tokens = um.get("total_tokens")
                if isinstance(um.get("input_tokens"), int) and input_tokens is None:
                    input_tokens = um.get("input_tokens")
                if isinstance(um.get("output_tokens"), int) and output_visible_tokens is None:
                    output_visible_tokens = um.get("output_tokens")

                itd = um.get("input_token_details")
                if isinstance(itd, dict) and isinstance(itd.get("cache_read"), int) and cache_history_tokens is None:
                    cache_history_tokens = itd.get("cache_read")

                otd = um.get("output_token_details")
                if isinstance(otd, dict) and isinstance(otd.get("reasoning"), int) and thoughts_tokens is None:
                    thoughts_tokens = otd.get("reasoning")

                if isinstance(total_tokens, int) and isinstance(input_tokens, int):
                    output_tokens = total_tokens - input_tokens

        # Some providers report only total/input tokens. In that case the
        # effective output tokens are the difference, which also matches the
        # game-level aggregation in llm_runtime.py.
        if output_tokens is None and isinstance(total_tokens, int) and isinstance(input_tokens, int):
            output_tokens = total_tokens - input_tokens

        return {
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thoughts_tokens": thoughts_tokens,
            "output_visible_tokens": output_visible_tokens,
            "cache_history_tokens": cache_history_tokens,
        }

    def _gemini_generate_with_cache(self, user_text: str) -> Tuple[str, str, Any]:
        return gemini_generate_with_cache_impl(self, user_text)

    def _gemini_generate_no_cache(self, user_text: str) -> Tuple[str, str, Any]:
        return gemini_generate_no_cache_impl(self, user_text)

    def _deepseek_generate(self, messages: List[Dict[str, str]]) -> Tuple[str, str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self._real_model,
            "messages": messages,
        }

        temp = self._generation_config.get("temperature", None)
        if temp is not None:
            kwargs["temperature"] = temp

        top_p = self._generation_config.get("top_p", None)
        if top_p is not None:
            kwargs["top_p"] = top_p

        max_tokens = self._generation_config.get("max_tokens", None)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        seed = self._generation_config.get("seed", None)
        if seed is not None:
            kwargs["seed"] = seed

        reasoning_effort = self._generation_config.get("reasoning_effort", None)
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        extra_body = self._generation_config.get("extra_body", None)
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        resp = self._deepseek_client.chat.completions.create(**kwargs)

        if resp is None:
            return "", "", resp

        choices = getattr(resp, "choices", None)
        if not isinstance(choices, list) or len(choices) == 0:
            return "", "", resp

        msg = getattr(choices[0], "message", None)
        if msg is None:
            return "", "", resp

        answer_text = getattr(msg, "content", "") or ""
        thoughts_text = getattr(msg, "reasoning_content", "") or ""

        return answer_text, thoughts_text, resp

    def send(self, text: str, step: Optional[int] = None) -> JsonDict:
        if not self._started:
            self.start()

        t0 = time.perf_counter()
        usage_for_metrics: Any = None

        is_game_prompt = not self._game_prompt_sent
        is_tool_result = "<TOOL_RESULT>" in text
        self._last_send_was_tool_result = is_tool_result

        if is_game_prompt:
            self._game_prompt_sent = True
            send_text = text
        elif is_tool_result:
            send_text = self._strip_knowledge_state_from_tool_result(text)
        else:
            send_text = text

        if self._print_tool_io and (is_game_prompt or is_tool_result):
            print("\n" + "=" * 60)
            print(
                "SENT TO MODEL MEMORY (GAME_PROMPT):"
                if is_game_prompt
                else "SENT TO MODEL MEMORY (TOOL_RESULT):"
            )
            print(send_text[:1000])
            print("=" * 60 + "\n")

        if self._require_context_cache and self._provider_spec.kind == "gemini":
            self._messages.append(self._HumanMessage(content=send_text))

            if self._cached_content_name:
                delta_msgs = self._messages[self._cached_message_count:-1]
                delta_text = self._messages_to_string(delta_msgs)
                if delta_text:
                    prompt_text = delta_text + "\n\n" + "USER:\n" + send_text
                else:
                    prompt_text = "USER:\n" + send_text
            else:
                prompt_text = self._format_transcript_for_cache()

            assistant_text, thoughts_text, gem_resp = self._gemini_generate_with_cache(prompt_text)
            self.print_model_raw(gem_resp)

            if isinstance(step, int):
                self.step_thought_texts[step] = thoughts_text or ""

            gem_usage = {}
            gm = getattr(gem_resp, "usage_metadata", None)
            if gm is None:
                gm = getattr(gem_resp, "usage", None)

            if gm is not None:
                if isinstance(gm, dict):
                    gem_usage = gm
                else:
                    try:
                        gem_usage = dict(gm)
                    except Exception:
                        try:
                            gem_usage = gm.__dict__
                        except Exception:
                            gem_usage = {"usage": str(gm)}

            if gem_usage:
                usage_for_metrics = gem_usage

            self._print_raw_model_text(assistant_text, tag="gemini_cached", step=step)
            self._messages.append(self._AIMessage(content=assistant_text))

            msgs_since_cache = len(self._messages) - self._cached_message_count
            can_create_first_cache = isinstance(step, int) and step >= self._first_cache_create_step

            if self._cached_content_name is None:
                if can_create_first_cache:
                    self._create_or_replace_cache()
            elif msgs_since_cache >= self._cache_update_threshold:
                self._create_or_replace_cache()

            out = self._from_text(assistant_text)
        else:
            self._messages.append(self._HumanMessage(content=send_text))

            if self._provider_spec.kind == "gemini":
                prompt_text = self._format_transcript_for_cache()
                raw_text, thoughts_text, resp = self._gemini_generate_no_cache(prompt_text)
                self.print_model_raw(resp)

                if isinstance(step, int):
                    self.step_thought_texts[step] = thoughts_text or ""

                gem_usage = {}
                gm = getattr(resp, "usage_metadata", None)
                if gm is None:
                    gm = getattr(resp, "usage", None)

                if gm is not None:
                    if isinstance(gm, dict):
                        gem_usage = gm
                    else:
                        try:
                            gem_usage = dict(gm)
                        except Exception:
                            try:
                                gem_usage = gm.__dict__
                            except Exception:
                                gem_usage = {"usage": str(gm)}

                if gem_usage:
                    usage_for_metrics = gem_usage

                self._print_raw_model_text(raw_text, tag="gemini_direct_no_cache", step=step)
                self._messages.append(self._AIMessage(content=raw_text))
                out = self._from_text(raw_text)
            elif self._provider == "deepseek":
                ds_messages = self._messages_to_openai_format(self._messages)
                raw_text, thoughts_text, resp = self._deepseek_generate(ds_messages)
                self.print_model_raw(resp)

                if isinstance(step, int):
                    self.step_thought_texts[step] = thoughts_text or ""

                usage_for_metrics = getattr(resp, "usage", None)

                if usage_for_metrics is not None:
                    pass

                self._print_raw_model_text(raw_text, tag="deepseek_direct", step=step)
                self._messages.append(self._AIMessage(content=raw_text))
                out = self._from_text(raw_text)
            else:
                resp = self._model.invoke(self._messages)
                self.print_model_raw(resp)

                usage = self._extract_usage_fields(resp)
                if usage:
                    usage_for_metrics = usage

                raw_text, thoughts_text = self._extract_openai_text_and_thoughts(resp)

                if isinstance(step, int):
                    self.step_thought_texts[step] = thoughts_text or ""

                self._print_raw_model_text(raw_text, tag="langchain", step=step)
                self._messages.append(self._AIMessage(content=raw_text))
                out = self._from_text(raw_text)
                out["thoughts"] = thoughts_text

        dt = time.perf_counter() - t0

        out["latency_s"] = dt

        cache_data = getattr(self, "cache_data", None)
        if not isinstance(cache_data, dict):
            cache_data = self.__dict__.get("cache_data")
        if isinstance(cache_data, dict):
            try:
                out["cache_data"] = cache_data
            except Exception:
                pass

        lat_dict = getattr(self, "_step_latencies", None)
        if isinstance(lat_dict, dict) and isinstance(step, int):
            lat_dict[step] = dt

        step_dict = getattr(self, "_step_tokens", None)
        if isinstance(step_dict, dict) and isinstance(step, int):
            tok = self._usage_to_metrics(usage_for_metrics)
            step_metrics = {
                "step_time_s": dt,
                "total_tokens": tok.get("total_tokens"),
                "input_tokens": tok.get("input_tokens"),
                "output_tokens": tok.get("output_tokens"),
                "thoughts_tokens": tok.get("thoughts_tokens"),
                "output_visible_tokens": tok.get("output_visible_tokens"),
                "cache_history_tokens": tok.get("cache_history_tokens"),
            }
            step_dict[step] = step_metrics

            # Launcher-facing log line. The Flask launcher parses this compact
            # JSON after every model response and shows the growing token usage
            # next to the corresponding chat step.
            try:
                print(
                    f"TOKEN_METRICS::STEP={step}::JSON="
                    f"{json.dumps(step_metrics, ensure_ascii=False, default=str)}",
                    flush=True,
                )
            except Exception:
                pass

        return out

    def _from_text(self, text: str) -> JsonDict:
        return from_text_impl(self, text)

    def _extract_tool_actions_from_text(self, text: str) -> List[JsonDict]:
        return extract_tool_actions_from_text_impl(self, text)