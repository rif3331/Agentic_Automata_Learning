"""
Responsible for:

- Managing the interactive LLM game runtime and chat-session execution loop
- Coordinating model prompts, tool handling, memory tracking, and step progression
- Tracking query behavior, token usage, latencies, and non-informative scoring
- Handling exports to CSV/HTML and producing runtime summaries and diagnostics
- Serving as the central orchestration layer connecting prompts, tools, models, and game state
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from game_types import JsonDict


class RunStoppedByUser(Exception):
    pass


STOP_REQUEST_FLAG_FILENAME = "STOP_REQUESTED.flag"


def _stop_request_flag_path() -> str:
    return os.path.join(os.getcwd(), STOP_REQUEST_FLAG_FILENAME)


def _raise_if_global_stop_requested() -> None:
    if os.path.exists(_stop_request_flag_path()):
        raise RunStoppedByUser("the run was stopped by the user")

from llm_exports import (
    export_run_to_csv as export_run_to_csv_impl,
    export_run_to_html as export_run_to_html_impl,
    read_existing_csv_header as read_existing_csv_header_impl,
    upgrade_csv_if_needed as upgrade_csv_if_needed_impl,
)
from llm_prompt_building import (
    build_followup_text as build_followup_text_impl,
    build_initial_text as build_initial_text_impl,
    build_last_bad_eq_block as build_last_bad_eq_block_impl,
    build_model_text as build_model_text_impl,
    get_initial_prompt_with_remaining_calls as get_initial_prompt_with_remaining_calls_impl,
    json_default as json_default_impl,
    print_sent_to_model as print_sent_to_model_impl,
    strip_knowledge_state_deep as strip_knowledge_state_deep_impl,
)
from llm_state_tracking import (
    build_noninformative_step_reason_map as build_noninformative_step_reason_map_impl,
    build_run_summary_dict as build_run_summary_dict_impl,
    collect_llm_mq_eq_counts as collect_llm_mq_eq_counts_impl,
    compute_llm_total_queries as compute_llm_total_queries_impl,
    dedup_by_first_tuple_item as dedup_by_first_tuple_item_impl,
    maybe_fail_on_score as maybe_fail_on_score_impl,
    snapshot_knowledge_state as snapshot_knowledge_state_impl,
    update_max_noninformative_score as update_max_noninformative_score_impl,
)
from passive_gold_runtime import (
    initialize_runtime_passive_gold,
    print_passive_learning_ui_snapshot,
    update_runtime_passive_gold_from_tool_reply,
)
from language_similarity_runtime import (
    initialize_runtime_language_similarity,
    print_language_similarity_ui_snapshot,
    update_runtime_language_similarity_from_eq_guesses,
)
from hypothesis_runtime import initialize_runtime_hypothesis_analysis
from llm_tool_handling import (
    extract_tool_calls as extract_tool_calls_impl,
    get_tool_by_name as get_tool_by_name_impl,
    handle_model_request as handle_model_request_impl,
    is_finished as is_finished_impl,
    limit_error_output as limit_error_output_impl,
    tool_call_limit_reached as tool_call_limit_reached_impl,
)


SAVE_FULL_LAST_MODEL_CONTEXT_TO_TXT = False
FULL_LAST_MODEL_CONTEXT_TXT_PATH = ""

@dataclass
class LLM_interactive_game:
    game: Any
    memory: List[JsonDict] = field(default_factory=list)

    _call_counter: int = field(default=0, init=False, repr=False)
    _last_oracle_response_call_number: Optional[int] = field(default=None, init=False, repr=False)
    _reached_optimal: bool = field(default=False, init=False, repr=False)
    _last_seed: Optional[int] = field(default=None, init=False, repr=False)
    _last_conversation_link: str = field(default="", init=False, repr=False)
    _last_llm_model_name: str = field(default="", init=False, repr=False)
    pending_tool_result: Optional[JsonDict] = None
    step_latencies: Dict[int, float] = field(default_factory=dict)
    step_tokens: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    eq_dfa_guesses: List[Tuple[int, Any, Optional[str]]] = field(default_factory=list)
    eq_duplicate_steps: List[Tuple[int, int]] = field(default_factory=list)
    mq_queries: List[Tuple[int, str, bool]] = field(default_factory=list)
    mq_duplicate_steps: List[Tuple[int, int]] = field(default_factory=list)
    mq_hits_previous_eq_witness: List[Tuple[int, int, str]] = field(default_factory=list)
    eq_contradicts_previous_mq: List[Tuple[int, int, str]] = field(default_factory=list)
    eq_contradicts_previous_eq_witness: List[Tuple[int, int, str]] = field(default_factory=list)

    noninformative_score: int = field(default=0, init=False)
    max_noninformative_score: int = field(default=0, init=False)
    _llm_failed: bool = field(default=False, init=False, repr=False)
    _stop_reason: str = field(default="", init=False, repr=False)

    _last_initial_prompt: str = field(default="", init=False, repr=False)
    _last_full_model_context_text: str = field(default="", init=False, repr=False)
    _run_crashed_mid_game: bool = field(default=False, init=False, repr=False)
    _run_crash_error: str = field(default="", init=False, repr=False)

    current_knowledge_state: Dict[str, Set[str]] = field(
        default_factory=lambda: {
            "words_accepted_by_dfa": set(),
            "words_rejected_by_dfa": set(),
        }
    )

    def __post_init__(self) -> None:
        for t in (getattr(self.game, "tools", None) or []):
            if getattr(t, "game", None) is None:
                setattr(t, "game", self.game)

    def _update_max_noninformative_score(self) -> None:
        update_max_noninformative_score_impl(self)

    def _print_sent_to_model(self, text: str, *, step: Optional[int] = None, tag: str = "") -> None:
        print_sent_to_model_impl(self, text, step=step, tag=tag)

    def _dedup_by_first_tuple_item(self, items: Any) -> List[Any]:
        return dedup_by_first_tuple_item_impl(items)

    def _build_noninformative_step_reason_map(self) -> Dict[int, str]:
        return build_noninformative_step_reason_map_impl(self)

    def build_run_summary_dict(self) -> Dict[str, Any]:
        return build_run_summary_dict_impl(self)

    def _exception_status_code(self, e: BaseException) -> Optional[int]:
        for attr in ("code", "status_code", "status"):
            value = getattr(e, attr, None)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)

        response = getattr(e, "response", None)
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value

        return None

    def _classify_model_send_error(self, e: BaseException) -> Tuple[bool, str]:
        status_code = self._exception_status_code(e)
        exc_type = type(e).__name__
        exc_module = type(e).__module__
        message = str(e).strip() or repr(e)
        combined = f"{exc_module}.{exc_type}: {message}"
        lower = combined.lower()

        programming_errors = (
            TypeError,
            ValueError,
            KeyError,
            AttributeError,
            ImportError,
            SyntaxError,
        )
        if isinstance(e, programming_errors):
            return (
                False,
                "Local code/configuration error. Waiting is unlikely to fix this.",
            )

        if status_code in {400, 401, 403, 404, 422}:
            return (
                False,
                f"Non-retryable API error HTTP {status_code}. The request, authentication, permissions, or model name should be fixed before retrying.",
            )

        non_retryable_markers = (
            "invalid_argument",
            "invalid api key",
            "api_key_invalid",
            "unauthenticated",
            "permission_denied",
            "permission denied",
            "not found",
            "model not found",
            "unsupported",
            "safety",
            "blocked",
            "malformed",
            "bad request",
        )
        if any(marker in lower for marker in non_retryable_markers):
            return (
                False,
                "Non-retryable request/auth/model error. Waiting 180 seconds is unlikely to change the result.",
            )

        connection_markers = (
            "connecterror",
            "connecttimeout",
            "connectionerror",
            "connection refused",
            "connection reset",
            "connection aborted",
            "network is unreachable",
            "name or service not known",
            "nodename nor servname",
            "getaddrinfo failed",
            "dns",
            "ssl",
            "certificate",
            "proxy",
        )
        if any(marker in lower for marker in connection_markers):
            return (
                False,
                "Connection/network problem while contacting the server. This usually requires fixing the network/proxy/certificate connection rather than waiting 180 seconds.",
            )

        retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
        if status_code in retryable_statuses:
            return (
                True,
                f"Retryable server/API error HTTP {status_code}.",
            )

        retryable_markers = (
            "resource_exhausted",
            "rate limit",
            "ratelimit",
            "quota",
            "too many requests",
            "overloaded",
            "unavailable",
            "temporarily unavailable",
            "deadline exceeded",
            "service unavailable",
            "internal server error",
            "bad gateway",
            "gateway timeout",
            "readtimeout",
            "timeout",
            "timed out",
            "server disconnected",
        )
        if any(marker in lower for marker in retryable_markers):
            return (
                True,
                "Retryable temporary server/rate-limit/timeout error.",
            )

        return (
            True,
            "Unknown model-provider error. Treating it as retryable by default.",
        )

    def _send_with_503_retry(self, chat_model: Any, text: str, *, step: int) -> JsonDict:
        attempt = 1
        while True:
            _raise_if_global_stop_requested()
            try:
                result = chat_model.send(text, step=step)
                _raise_if_global_stop_requested()
                return result
            except RunStoppedByUser:
                raise
            except Exception as e:
                should_retry, reason = self._classify_model_send_error(e)
                exc_type = type(e).__name__
                message = str(e).strip() or repr(e)

                if should_retry:
                    print(
                        "Server/model request error. "
                        f"Attempt {attempt} failed at agent tool call #{step}. "
                        f"Decision: waiting 180 seconds and then retrying. "
                        f"Reason: {reason} "
                        f"Error type: {exc_type}. Error: {message}"
                    )
                    for _ in range(180):
                        _raise_if_global_stop_requested()
                        time.sleep(1)
                    attempt += 1
                    continue

                print(
                    "Server/model request error. "
                    f"Attempt {attempt} failed at agent tool call #{step}. "
                    "Decision: exiting without waiting. "
                    f"Reason: {reason} "
                    f"Error type: {exc_type}. Error: {message}"
                )
                raise

    def run_with_chat_session(
        self,
        chat_model: Any,
        max_steps: int = 1000,
        verbose: bool = True,
        *,
        export_csv_path: Optional[str] = None,
        export_html_path: Optional[str] = None,
        seed: Optional[int] = None,
        conversation_link: str = "",
    ) -> None:
        if not getattr(chat_model, "send", None):
            raise TypeError("chat_model must have send(text)->dict")

        game_t0 = time.perf_counter()

        if not hasattr(self, "step_latencies") or not isinstance(getattr(self, "step_latencies"), dict):
            self.step_latencies = {}

        if not hasattr(self, "step_tokens") or not isinstance(getattr(self, "step_tokens"), dict):
            self.step_tokens = {}

        self.step_latencies.clear()
        self.step_tokens.clear()

        setattr(chat_model, "_step_latencies", self.step_latencies)
        setattr(chat_model, "_step_tokens", self.step_tokens)

        self._last_llm_model_name = str(getattr(chat_model, "model_name", "") or "")
        self._last_seed = seed
        self._last_conversation_link = conversation_link or ""
        self._last_game_mode = "chat"
        last_full_model_context_text = ""
        self._run_crashed_mid_game = False
        self._run_crash_error = ""

        stop_request_flag_path = _stop_request_flag_path()
        try:
            if os.path.exists(stop_request_flag_path):
                os.remove(stop_request_flag_path)
        except Exception:
            pass

        def _check_stop_requested() -> None:
            if os.path.exists(stop_request_flag_path):
                try:
                    os.remove(stop_request_flag_path)
                except Exception:
                    pass
                self._run_crashed_mid_game = True
                self._run_crash_error = "StoppedByUser: the run was stopped by the user"
                raise RunStoppedByUser("the run was stopped by the user")

        initialize_runtime_passive_gold(self)
        initialize_runtime_language_similarity(self)
        initialize_runtime_hypothesis_analysis(self)

        def _build_full_model_context_text(current_text: str) -> str:
            send_text = current_text

            if "<TOOL_RESULT>" in send_text:
                strip_fn = getattr(chat_model, "_strip_knowledge_state_from_tool_result", None)
                if callable(strip_fn):
                    send_text = strip_fn(send_text)

            previous_context = ""
            format_fn = getattr(chat_model, "_format_transcript_for_cache", None)
            if callable(format_fn):
                try:
                    previous_context = format_fn()
                except Exception:
                    previous_context = ""

            if not previous_context:
                messages_to_string_fn = getattr(chat_model, "_messages_to_string", None)
                messages = getattr(chat_model, "_messages", None)
                if callable(messages_to_string_fn) and messages is not None:
                    try:
                        previous_context = messages_to_string_fn(messages)
                    except Exception:
                        previous_context = ""

            parts = []
            if previous_context:
                parts.append(previous_context)
            parts.append("USER:\n" + send_text)
            return "\n\n".join(parts).strip()

        def _write_full_last_model_context_file(text: str) -> None:
            if not SAVE_FULL_LAST_MODEL_CONTEXT_TO_TXT:
                return
            if not text:
                return

            output_path = FULL_LAST_MODEL_CONTEXT_TXT_PATH

            if not output_path:
                base_dir = ""
                if export_html_path:
                    base_dir = os.path.dirname(os.path.abspath(export_html_path))
                elif export_csv_path:
                    base_dir = os.path.dirname(os.path.abspath(export_csv_path))
                else:
                    base_dir = os.getcwd()

                seed_part = f"_seed_{seed}" if seed is not None else ""
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(
                    base_dir,
                    f"full_last_model_context{seed_part}_{timestamp}.txt",
                )

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            self._last_full_model_context_txt_path = output_path
            print(f"FULL LAST MODEL CONTEXT SAVED TO: {output_path}")

        def _extract_tool_reply_call_number(tool_reply: JsonDict) -> Optional[int]:
            outs = tool_reply.get("tool_outputs") if isinstance(tool_reply, dict) else None
            if isinstance(outs, list) and outs:
                first = outs[0]
                if isinstance(first, dict):
                    call_count = first.get("call_count")
                    if isinstance(call_count, int) and call_count > 0:
                        return call_count
            return None

        def _print_step(step_i: int, sent: str) -> None:
            if not verbose:
                return

            budget = getattr(self.game, "max_tool_calls", 0)
            budget_str = str(budget) if (isinstance(budget, int) and budget > 0) else "∞"
            remaining_calls = (
                str(max(0, int(budget) - self._call_counter))
                if (isinstance(budget, int) and budget > 0)
                else "∞"
            )
            print(
                f"MODEL INPUT METADATA | remaining_calls={remaining_calls} | "
                f"calls_used={self._call_counter}/{budget_str} | input chars={len(sent)}"
            )

        try:
            _check_stop_requested()
            initial_text = self._build_model_text(include_knowledge_state=False)
            self.memory.append({"role": "user", "content": initial_text, "raw": {"sent": initial_text}})
            _print_step(0, initial_text)

            last_full_model_context_text = _build_full_model_context_text(initial_text)
            self._print_sent_to_model(initial_text, step=None, tag="CHAT_INIT")
            model_out = self._send_with_503_retry(chat_model, initial_text, step=self._call_counter + 1)
            _check_stop_requested()

            self.memory.append({"role": "assistant", "content": model_out.get("content", ""), "raw": model_out})

            if verbose:
                c = model_out.get("content", "") or ""
                if "<TOOL_ACTION>" not in c:
                    print("MODEL:", (c[:600] + ("..." if len(c) > 600 else "")))

            for step in range(1, max_steps + 1):
                _check_stop_requested()
                tool_reply = self.handle_model_request(model_out)
                _check_stop_requested()
                self.memory.append({"role": "tool", "content": "", "raw": tool_reply})
                # Print the passive-learning UI card BEFORE adding the current
                # oracle answer to the passive observations. This makes the card
                # show exactly the knowledge available before this query: the
                # first card is empty, and every following card is shifted one
                # query back.
                print_passive_learning_ui_snapshot(self, tool_reply)
                update_runtime_passive_gold_from_tool_reply(self, tool_reply)
                update_runtime_language_similarity_from_eq_guesses(self)
                print_language_similarity_ui_snapshot(self, tool_reply)

                # Print only the formatted oracle response block, including the final one.
                # Do not print the raw Python dict form.
                followup = self._build_followup_text(tool_reply)
                self._last_oracle_response_call_number = _extract_tool_reply_call_number(tool_reply)
                if verbose:
                    self._print_sent_to_model(followup, step=None, tag="CHAT_TOOL_RESULT")

                if self._is_finished(model_out, tool_reply):
                    if verbose and getattr(self, "_llm_failed", False):
                        print(f"GAME STOPPED REASON: {self._stop_reason or 'LLM_FAILED'}")
                    break

                self.memory.append({"role": "user", "content": followup, "raw": {"sent": followup}})
                _print_step(step, followup)
                last_full_model_context_text = _build_full_model_context_text(followup)
                model_out = self._send_with_503_retry(chat_model, followup, step=self._call_counter + 1)
                _check_stop_requested()
                self.memory.append({"role": "assistant", "content": model_out.get("content", ""), "raw": model_out})

                if verbose:
                    c = model_out.get("content", "") or ""
                    if "<TOOL_ACTION>" not in c:
                        print("MODEL:", (c[:600] + ("..." if len(c) > 600 else "")))

            return

        except RunStoppedByUser:
            self._run_crashed_mid_game = True
            if not self._run_crash_error:
                self._run_crash_error = "StoppedByUser: the run was stopped by the user"

        except BaseException as e:
            self._run_crashed_mid_game = True
            self._run_crash_error = f"{type(e).__name__}: {e}"
            raise

        finally:
            game_dt = time.perf_counter() - game_t0
            self._last_game_time_s = game_dt

            total_tokens = 0
            input_tokens = 0
            cache_history_tokens = 0

            for _k, v in (self.step_tokens or {}).items():
                if not isinstance(v, dict):
                    continue

                tt = v.get("total_tokens")
                it = v.get("input_tokens")

                ct = (
                    v.get("prompt_cache_hit_tokens")
                    or v.get("cached_tokens")
                    or v.get("cache_history_tokens")
                    or 0
                )

                if isinstance(tt, int):
                    total_tokens += tt
                if isinstance(it, int):
                    input_tokens += it
                if isinstance(ct, int):
                    cache_history_tokens += ct

            output_tokens = total_tokens - input_tokens
            total_tokens_no_cache = total_tokens - cache_history_tokens

            self._last_game_total_tokens = total_tokens
            self._last_game_token_tuple = {
                "total": total_tokens,
                "in": input_tokens,
                "out": output_tokens,
                "cache_hit": cache_history_tokens,
                "total_no_cache": total_tokens_no_cache,
            }

            cache_stats = getattr(chat_model, "cache_stats", None)
            self._last_cache_stats = cache_stats if isinstance(cache_stats, dict) else {}

            cache_data = getattr(chat_model, "cache_data", None)
            step_thought_texts_obj = getattr(chat_model, "step_thought_texts", None)
            self._last_step_thought_texts = (
                step_thought_texts_obj if isinstance(step_thought_texts_obj, dict) else {}
            )

            if not isinstance(cache_data, dict):
                cache_data = getattr(chat_model, "__dict__", {}).get("נתוני cache")

            self._last_cache_data = (
                cache_data
                if isinstance(cache_data, dict)
                else {"cache_count": 0, "total_tokens": 0, "token_counts": []}
            )

            self._last_full_model_context_text = last_full_model_context_text or ""

            if os.path.exists(stop_request_flag_path):
                try:
                    os.remove(stop_request_flag_path)
                except Exception:
                    pass
                self._run_crashed_mid_game = True
                self._run_crash_error = "StoppedByUser: the run was stopped by the user"

            # Do not write the full last model context to a separate .txt file.
            # The same text is exported into the CSV by llm_exports.py.

            if export_html_path:
                self.export_run_to_html(export_html_path, seed=seed, conversation_link=conversation_link)

            print("\n" + "=" * 60)
            print("Game summary")
            print("=" * 60)
            print(f"TOTAL GAME TIME: {game_dt:.4f}s")
            if getattr(self, "_run_crashed_mid_game", False):
                print("GAME CRASHED - PARTIAL RESULT SAVED")
                print(f"Crash: {getattr(self, '_run_crash_error', '') or 'Unknown error'}")
            elif getattr(self, "_reached_optimal", False):
                print("GAME FINISHED - LLM WON")
            else:
                print("GAME FINISHED - LLM LOST")

            if export_html_path:
                try:
                    from pathlib import Path

                    html_link = Path(os.path.abspath(export_html_path)).as_uri()
                except Exception:
                    html_link = os.path.abspath(export_html_path)

                print(
                    "Visual game display: click here to view it: "
                    f"{html_link}"
                )

            # Keep the CSV message as the final line printed at the end of the run.
            if export_csv_path:
                self.export_run_to_csv(export_csv_path, seed=seed, conversation_link=conversation_link)

    def _compute_llm_total_queries(self) -> Any:
        return compute_llm_total_queries_impl(self)

    def _get_strategies_summary(self) -> str:
        dfa = getattr(self.game, "dfa", None)
        if dfa is None:
            return ""

        runs = getattr(dfa, "strategy_results", None)
        if not isinstance(runs, list) or not runs:
            return ""

        last = runs[-1]
        if not isinstance(last, dict):
            return ""

        parts: List[str] = []
        for strat_name, res in last.items():
            total = getattr(res, "total_queries", None)
            if isinstance(total, int):
                parts.append(f"{strat_name}={total}")

        return ";".join(parts)

    def _read_existing_csv_header(self, csv_path: str) -> Optional[List[str]]:
        return read_existing_csv_header_impl(csv_path)

    def _upgrade_csv_if_needed(self, csv_path: str, new_header: List[str]) -> None:
        upgrade_csv_if_needed_impl(csv_path, new_header)

    def export_run_to_csv(
        self,
        csv_path: str,
        *,
        seed: Optional[int] = None,
        conversation_link: str = "",
    ) -> None:
        export_run_to_csv_impl(self, csv_path, seed=seed, conversation_link=conversation_link)

    def _snapshot_knowledge_state(self, ks: Any) -> Dict[str, Set[str]]:
        return snapshot_knowledge_state_impl(ks)

    def _maybe_fail_on_score(self) -> None:
        maybe_fail_on_score_impl(self)

    def handle_model_request(self, model_output: JsonDict) -> JsonDict:
        return handle_model_request_impl(self, model_output)

    def _build_initial_text(self) -> str:
        return build_initial_text_impl(self)

    def get_initial_prompt_with_remaining_calls(self) -> str:
        return get_initial_prompt_with_remaining_calls_impl(self)

    def _json_default(self, o: Any) -> Any:
        return json_default_impl(self, o)

    def _build_last_bad_eq_block(self) -> str:
        return build_last_bad_eq_block_impl(self)

    def _build_model_text(self, *, include_knowledge_state: bool = True) -> str:
        return build_model_text_impl(self, include_knowledge_state=include_knowledge_state)

    def _strip_knowledge_state_deep(self, obj: Any) -> Any:
        return strip_knowledge_state_deep_impl(self, obj)

    def _build_followup_text(self, tool_reply: JsonDict) -> str:
        return build_followup_text_impl(self, tool_reply)

    def _tool_call_limit_reached(self) -> bool:
        return tool_call_limit_reached_impl(self)

    def _limit_error_output(self):
        return limit_error_output_impl(self)

    def _get_tool_by_name(self, requested_name: Any):
        return get_tool_by_name_impl(self, requested_name)

    def _extract_tool_calls(self, model_output: JsonDict):
        return extract_tool_calls_impl(model_output)

    def _is_finished(self, model_output: JsonDict, tool_reply: JsonDict) -> bool:
        return is_finished_impl(self, model_output, tool_reply)

    def _collect_llm_mq_eq_counts(self) -> Tuple[int, int]:
        return collect_llm_mq_eq_counts_impl(self)

    def export_run_to_html(
        self,
        html_path: str,
        *,
        seed: Optional[int] = None,
        conversation_link: str = "",
    ) -> None:
        export_run_to_html_impl(self, html_path, seed=seed, conversation_link=conversation_link)