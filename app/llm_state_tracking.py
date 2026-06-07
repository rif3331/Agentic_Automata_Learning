"""
Responsible for:

- Tracking and scoring non-informative behavior during interaction
- Building run summaries and per-step non-informative reason mappings
- Managing knowledge-state snapshots and score-based failure conditions
- Computing total LLM query counts and separating MQ vs EQ usage statistics
- Supporting diagnostic reporting for runtime behavior and stopping analysis
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple
from constants import NONINFORMATIVE_SCORE_STOP
from utils import normalize_tool_name
from game_types import EvaluationToolInterface


def dedup_by_first_tuple_item(items: Any) -> List[Any]:
    if not isinstance(items, list):
        return []

    out: List[Any] = []
    seen: Set[Any] = set()

    for t in items:
        if not isinstance(t, tuple) or len(t) == 0:
            continue
        k = t[0]
        if k in seen:
            continue
        seen.add(k)
        out.append(t)

    return out


def update_max_noninformative_score(owner: Any) -> None:
    if owner.noninformative_score > owner.max_noninformative_score:
        owner.max_noninformative_score = owner.noninformative_score


def build_noninformative_step_reason_map(owner: Any) -> Dict[int, str]:
    reasons: Dict[int, Dict[str, str]] = {}

    def add(step: Any, kind: str, cause_step: Any) -> None:
        try:
            st = int(step)
            cs = int(cause_step)
        except Exception:
            return

        if st not in reasons:
            reasons[st] = {}

        if kind in reasons[st]:
            return

        reasons[st][kind] = f"{kind.lower()}({cs})"

    for cur_step, prev_step in list(owner.mq_duplicate_steps):
        add(cur_step, "MQ DUPLICATE STEPS", prev_step)

    for cur_step, hit_eq_step, _w in list(owner.mq_hits_previous_eq_witness):
        add(cur_step, "MQ HITS PREVIOUS EQ WITNESS", hit_eq_step)

    for cur_step, prev_step in list(owner.eq_duplicate_steps):
        add(cur_step, "EQ DUPLICATE STEPS", prev_step)

    for cur_step, mq_step, _mq_word in dedup_by_first_tuple_item(list(owner.eq_contradicts_previous_mq)):
        add(cur_step, "EQ CONTRADICTS PREVIOUS MQ", mq_step)

    for cur_step, prev_eq_step, _prev_witness in dedup_by_first_tuple_item(list(owner.eq_contradicts_previous_eq_witness)):
        add(cur_step, "EQ CONTRADICTS PREVIOUS EQ WITNESS", prev_eq_step)

    out: Dict[int, str] = {}
    for st, m in reasons.items():
        out[st] = " | ".join(m.values())

    return out


def build_run_summary_dict(owner: Any) -> Dict[str, Any]:
    return {
        "MQ DUPLICATE STEPS": list(owner.mq_duplicate_steps),
        "MQ HITS PREVIOUS EQ WITNESS": list(owner.mq_hits_previous_eq_witness),
        "EQ DUPLICATE STEPS": list(owner.eq_duplicate_steps),
        "EQ CONTRADICTS PREVIOUS MQ": dedup_by_first_tuple_item(list(owner.eq_contradicts_previous_mq)),
        "EQ CONTRADICTS PREVIOUS EQ WITNESS": dedup_by_first_tuple_item(list(owner.eq_contradicts_previous_eq_witness)),
        "NONINFORMATIVE SCORE LAST": int(owner.noninformative_score),
        "NONINFORMATIVE SCORE MAX": int(owner.max_noninformative_score),
    }


def snapshot_knowledge_state(ks: Any) -> Dict[str, set[str]]:
    if not isinstance(ks, dict):
        return {
            "words_accepted_by_dfa": set(),
            "words_rejected_by_dfa": set(),
        }

    acc = ks.get("words_accepted_by_dfa")
    rej = ks.get("words_rejected_by_dfa")

    return {
        "words_accepted_by_dfa": set(acc) if isinstance(acc, set) else set(),
        "words_rejected_by_dfa": set(rej) if isinstance(rej, set) else set(),
    }


def maybe_fail_on_score(owner: Any) -> None:
    try:
        stop_at = int(NONINFORMATIVE_SCORE_STOP)
    except Exception:
        stop_at = 100

    if owner.noninformative_score >= stop_at:
        owner._llm_failed = True
        if not owner._stop_reason:
            owner._stop_reason = (
                f"NONINFORMATIVE_SCORE_STOP_REACHED "
                f"score={owner.noninformative_score} stop={stop_at}"
            )


def compute_llm_total_queries(owner: Any) -> Any:
    last_ok_call: Optional[int] = None

    for m in reversed(owner.memory):
        if m.get("role") != "tool":
            continue

        raw = m.get("raw")
        if not isinstance(raw, dict):
            continue

        outs = raw.get("tool_outputs")
        if not isinstance(outs, list) or not outs:
            continue

        for out in reversed(outs):
            if not isinstance(out, dict):
                continue
            if out.get("error") is not None:
                continue
            if str(out.get("tool_name", "")).strip().lower() in ("tool_budget", "toolbudget"):
                continue

            cc = out.get("call_count")
            if isinstance(cc, int) and cc > 0:
                last_ok_call = cc
                break

        if last_ok_call is not None:
            break

    return last_ok_call if last_ok_call is not None else owner._call_counter


def collect_llm_mq_eq_counts(owner: Any) -> Tuple[int, int]:
    mq = 0
    eq = 0

    eval_tool_names = set()
    for t in (getattr(owner.game, "tools", None) or []):
        name = getattr(t, "tool_name", t.__class__.__name__)
        try:
            if isinstance(t, EvaluationToolInterface):
                eval_tool_names.add(normalize_tool_name(name))
        except Exception:
            pass

    for m in owner.memory:
        if m.get("role") != "tool":
            continue

        raw = m.get("raw")
        if not isinstance(raw, dict):
            continue

        outs = raw.get("tool_outputs", [])
        if not isinstance(outs, list):
            continue

        for out in outs:
            if not isinstance(out, dict):
                continue
            tn = normalize_tool_name(out.get("tool_name"))
            if tn == "is_word_in_language":
                mq += 1
            elif tn in eval_tool_names:
                eq += 1

    return mq, eq