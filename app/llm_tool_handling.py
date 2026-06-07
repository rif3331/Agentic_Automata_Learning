"""
Responsible for:

- Handling model-generated tool calls and dispatching tool execution
- Enforcing tool-call limits, completion checks, and tool lookup resolution
- Managing knowledge-state updates from tool outputs
- Tracking duplicate, contradictory, and non-informative MQ/EQ behavior
- Updating scoring, failure conditions, and interaction stopping logic
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from constants import (
    NONINF_EQ_CONTRADICTS_PREV_EQ_WITNESS_POINTS,
    NONINF_EQ_CONTRADICTS_PREV_MQ_POINTS,
    NONINF_EQ_DUPLICATE_POINTS,
    NONINF_MQ_DUPLICATE_POINTS,
    NONINF_MQ_HIT_EQ_WITNESS_POINTS,
)
from game_types import EvaluationToolInterface, ToolInput, ToolInterface, ToolOutput, JsonDict
from utils import word_to_dfa_input, canonical_word, normalize_tool_name
from hypothesis_runtime import update_runtime_hypothesis_analysis_from_candidate

_TOOL_ACTION_RE = re.compile(r"<TOOL_ACTION>\s*(\{.*?\})\s*</TOOL_ACTION>", re.DOTALL)


def tool_call_limit_reached(owner: Any) -> bool:
    m = getattr(owner.game, "max_tool_calls", 0)
    return isinstance(m, int) and m > 0 and owner._call_counter >= m


def limit_error_output(owner: Any) -> ToolOutput:
    return {
        "tool_name": "tool_budget",
        "call_count": owner._call_counter,
        "error": "MAX_TOOL_CALLS_REACHED",
        "output": {"max_tool_calls": getattr(owner.game, "max_tool_calls", 0)},
        "knowledge_state": owner.current_knowledge_state,
    }


def get_tool_by_name(owner: Any, requested_name: Any) -> Optional[ToolInterface]:
    req = normalize_tool_name(requested_name)

    for t in (getattr(owner.game, "tools", None) or []):
        name = getattr(t, "tool_name", t.__class__.__name__)
        if normalize_tool_name(name) == req:
            return t

    return None


def extract_tool_calls(model_output: JsonDict) -> List[Tuple[str, Dict[str, Any]]]:
    calls: List[Tuple[str, Dict[str, Any]]] = []

    tcs = model_output.get("tool_calls")
    if isinstance(tcs, list):
        for item in tcs:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                calls.append((item["name"], item.get("args", {}) or {}))

    text = model_output.get("content", "")
    if isinstance(text, str) and text:
        for m in _TOOL_ACTION_RE.finditer(text):
            raw = m.group(1)
            try:
                obj = json.loads(raw)
            except Exception:
                continue

            name = obj.get("tool_name")
            args = obj.get("input", {})
            if isinstance(name, str) and name.strip():
                calls.append((name, args if isinstance(args, dict) else {}))

    return calls




def _set_noninformative_analysis(out: ToolOutput, *, is_noninformative: bool, kind: str = "", details: str = "") -> None:
    """Kept for compatibility with older call sites.

    Non-informative-query diagnostics are intentionally not attached to the
    oracle TOOL_RESULT, because TOOL_RESULT is sent back to the model. The
    launcher reads these diagnostics from the separate NONINFORMATIVE_ANALYSIS
    log line printed by _print_noninformative_analysis.
    """
    return

def _print_noninformative_analysis(call_count: int, *, is_noninformative: bool, kind: str = "", details: str = "") -> None:
    status = "YES" if is_noninformative else "NO"
    print(
        f"NONINFORMATIVE_ANALYSIS::CALL={call_count}::STATUS={status}::TYPE={kind or '-'}::DETAILS={details or '-'}",
        flush=True,
    )

def is_finished(owner: Any, model_output: JsonDict, tool_reply: JsonDict) -> bool:
    if getattr(owner, "_llm_failed", False):
        return True

    outs = tool_reply.get("tool_outputs", [])
    if isinstance(outs, list):
        for out in outs:
            if isinstance(out, dict):
                payload = out.get("output") or {}
                if isinstance(payload, dict) and payload.get("optimal") is True:
                    return True

    if owner._tool_call_limit_reached():
        return True

    if model_output.get("finish") is True:
        return True

    return False


def handle_model_request(owner: Any, model_output: JsonDict) -> JsonDict:
    calls = owner._extract_tool_calls(model_output)

    if not calls:
        return {"error": "NO_TOOL_CALLS_DETECTED"}

    name, args = calls[0]

    if owner._tool_call_limit_reached():
        out = owner._limit_error_output()
        out["knowledge_state"] = owner._snapshot_knowledge_state(owner.current_knowledge_state)
        return {"tool_outputs": [out]}

    tool = owner._get_tool_by_name(name)

    if tool is None:
        return {"error": "NO_TOOL_CALLS_DETECTED"}

    # Reserve the next tool-call number, but do not consume the budget yet.
    # The budget is consumed only after the oracle returns a successful tool output.
    call_count = int(getattr(owner, "_call_counter", 0)) + 1

    req: ToolInput = {
        "tool_name": getattr(tool, "tool_name", tool.__class__.__name__),
        "call_count": call_count,
        "input": args if isinstance(args, dict) else {},
        "knowledge_state": owner.current_knowledge_state,
    }
    out = tool.invoke(req)

    # Invalid tool inputs or any tool-level error should not consume a tool-call budget.
    # Treat them the same way as a missing/undetected tool call so the agent can retry.
    if isinstance(out, dict) and out.get("error") is not None:
        return {"error": "NO_TOOL_CALLS_DETECTED"}

    # Count the tool call only after the oracle accepted and answered it successfully.
    owner._call_counter = call_count

    ks_out = out.get("knowledge_state")
    if isinstance(ks_out, dict):
        owner.current_knowledge_state = ks_out

    out["knowledge_state"] = owner._snapshot_knowledge_state(out.get("knowledge_state"))

    tool_name_norm = normalize_tool_name(out.get("tool_name"))

    dfa_obj = getattr(owner.game, "dfa", None)
    alphabet = getattr(dfa_obj, "input_symbols", None) if dfa_obj is not None else None

    if tool_name_norm == "is_word_in_language":
        payload = out.get("output") or {}
        mq_is_duplicate = False
        mq_hits_prev_eq_witness = False
        mq_duplicate_prev_step: Optional[int] = None

        if isinstance(payload, dict):
            w_raw = payload.get("word")
            w = canonical_word(w_raw, alphabet)
            payload["word"] = w
            acc = payload.get("accepted")
            hit_eq_step: Optional[int] = None

            for eq_step, _cand, witness in reversed(owner.eq_dfa_guesses):
                if isinstance(witness, str) and witness == w and witness != "":
                    hit_eq_step = int(eq_step)
                    break

            if hit_eq_step is not None and isinstance(w, str):
                owner.mq_hits_previous_eq_witness.append((call_count, hit_eq_step, w))
                mq_hits_prev_eq_witness = True

            if isinstance(w, str) and isinstance(acc, bool):
                current_step = call_count
                prev_step: Optional[int] = None

                for s, ww, _aa in owner.mq_queries:
                    if ww == w:
                        prev_step = int(s)
                        break

                if prev_step is not None:
                    owner.mq_duplicate_steps.append((current_step, prev_step))
                    mq_is_duplicate = True
                    mq_duplicate_prev_step = prev_step

                owner.mq_queries.append((current_step, w, acc))

        if mq_is_duplicate:
            owner.noninformative_score += int(NONINF_MQ_DUPLICATE_POINTS)
        elif mq_hits_prev_eq_witness:
            owner.noninformative_score += int(NONINF_MQ_HIT_EQ_WITNESS_POINTS)
        else:
            owner.noninformative_score = 0

        if mq_is_duplicate:
            _set_noninformative_analysis(
                out,
                is_noninformative=True,
                kind="Repeated query",
                details=f"Repeated MQ from tool call #{mq_duplicate_prev_step}",
            )
            _print_noninformative_analysis(
                call_count,
                is_noninformative=True,
                kind="Repeated query",
                details=f"Repeated MQ from tool call #{mq_duplicate_prev_step}",
            )
        else:
            _set_noninformative_analysis(out, is_noninformative=False)
            _print_noninformative_analysis(call_count, is_noninformative=False)

        owner._update_max_noninformative_score()
        owner._maybe_fail_on_score()

    if isinstance(tool, EvaluationToolInterface):
        payload = out.get("output") or {}
        if isinstance(payload, dict):
            cand_obj = payload.get("_candidate_obj")

            if cand_obj is not None:
                current_eq_step = call_count
                prev_eq_step: Optional[int] = None
                eq_is_duplicate = False
                eq_contra_prev_mq_step: Optional[int] = None
                eq_contra_prev_mq_word: Optional[str] = None

                for prev_step, prev_dfa, _prev_witness in owner.eq_dfa_guesses:
                    try:
                        ok, _w = cand_obj.eq(prev_dfa)
                    except Exception:
                        ok = False

                    if ok:
                        prev_eq_step = int(prev_step)
                        break

                if prev_eq_step is not None:
                    owner.eq_duplicate_steps.append((current_eq_step, prev_eq_step))
                    eq_is_duplicate = True

                before_mq_contra = len(owner.eq_contradicts_previous_mq)
                for mq_step, mq_word, mq_acc in owner.mq_queries:
                    inp = word_to_dfa_input(mq_word, alphabet)
                    try:
                        cand_accepts = bool(cand_obj.accepts_input(inp))
                    except Exception:
                        continue

                    if mq_acc and (not cand_accepts):
                        owner.eq_contradicts_previous_mq.append((current_eq_step, int(mq_step), mq_word))
                        if eq_contra_prev_mq_step is None:
                            eq_contra_prev_mq_step = int(mq_step)
                            eq_contra_prev_mq_word = str(mq_word)
                    elif (not mq_acc) and cand_accepts:
                        owner.eq_contradicts_previous_mq.append((current_eq_step, int(mq_step), mq_word))
                        if eq_contra_prev_mq_step is None:
                            eq_contra_prev_mq_step = int(mq_step)
                            eq_contra_prev_mq_word = str(mq_word)

                eq_contra_prev_mq = len(owner.eq_contradicts_previous_mq) > before_mq_contra

                before_eqw_contra = len(owner.eq_contradicts_previous_eq_witness)
                original_dfa = getattr(owner.game, "dfa", None)
                for prev_eq_step_src, _prev_cand, prev_witness in owner.eq_dfa_guesses:
                    if not isinstance(prev_witness, str) or prev_witness == "":
                        continue

                    inp_w = word_to_dfa_input(prev_witness, alphabet)
                    try:
                        cand_val = bool(cand_obj.accepts_input(inp_w))
                    except Exception:
                        continue

                    true_val: Optional[bool] = None
                    if original_dfa is not None:
                        if hasattr(original_dfa, "accepts_input"):
                            try:
                                true_val = bool(original_dfa.accepts_input(inp_w))
                            except Exception:
                                true_val = None
                        elif hasattr(original_dfa, "mq"):
                            try:
                                true_val = bool(original_dfa.mq(prev_witness))
                            except Exception:
                                true_val = None

                    if isinstance(true_val, bool) and cand_val != true_val:
                        owner.eq_contradicts_previous_eq_witness.append(
                            (current_eq_step, int(prev_eq_step_src), prev_witness)
                        )

                eq_contra_prev_eq_witness = (
                    len(owner.eq_contradicts_previous_eq_witness) > before_eqw_contra
                )

                witness_raw = payload.get("witness_word")
                witness = canonical_word(witness_raw, alphabet) if isinstance(witness_raw, str) else None
                payload["witness_word"] = witness
                owner.eq_dfa_guesses.append((current_eq_step, cand_obj, witness))
                update_runtime_hypothesis_analysis_from_candidate(
                    owner,
                    step=current_eq_step,
                    hypothesis_dfa=cand_obj,
                )

                if eq_is_duplicate:
                    owner.noninformative_score += int(NONINF_EQ_DUPLICATE_POINTS)
                elif eq_contra_prev_mq:
                    owner.noninformative_score += int(NONINF_EQ_CONTRADICTS_PREV_MQ_POINTS)
                elif eq_contra_prev_eq_witness:
                    owner.noninformative_score += int(NONINF_EQ_CONTRADICTS_PREV_EQ_WITNESS_POINTS)
                else:
                    owner.noninformative_score = 0

                if eq_is_duplicate:
                    _set_noninformative_analysis(
                        out,
                        is_noninformative=True,
                        kind="Repeated query",
                        details=f"Repeated EQ from tool call #{prev_eq_step}",
                    )
                    _print_noninformative_analysis(
                        call_count,
                        is_noninformative=True,
                        kind="Repeated query",
                        details=f"Repeated EQ from tool call #{prev_eq_step}",
                    )
                elif eq_contra_prev_mq:
                    _set_noninformative_analysis(
                        out,
                        is_noninformative=True,
                        kind="Contradiction of previous information",
                        details=f"Hypothesis contradicts MQ #{eq_contra_prev_mq_step} on word {eq_contra_prev_mq_word}",
                    )
                    _print_noninformative_analysis(
                        call_count,
                        is_noninformative=True,
                        kind="Contradiction of previous information",
                        details=f"Hypothesis contradicts MQ #{eq_contra_prev_mq_step} on word {eq_contra_prev_mq_word}",
                    )
                else:
                    _set_noninformative_analysis(out, is_noninformative=False)
                    _print_noninformative_analysis(call_count, is_noninformative=False)

                owner._update_max_noninformative_score()
                owner._maybe_fail_on_score()

            payload.pop("_candidate_obj", None)

            if payload.get("optimal") is True:
                owner._reached_optimal = True

    if getattr(owner, "_llm_failed", False) and owner._stop_reason:
        print(f"GAME STOPPED REASON: {owner._stop_reason}")

    return {"tool_outputs": [out]}