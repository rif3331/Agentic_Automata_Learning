"""
Runtime passive-learner analysis.

This module computes passive-learner first-success columns during the run.
The LLM part is updated incrementally after each accepted tool call, so it is
not recomputed from scratch when exporting CSV/HTML.

It intentionally does NOT build observation tables and does NOT compute the
LLM/LStar/TTT representative summaries.
"""
from __future__ import annotations

import html
import json
import queue
import threading
from typing import Any, Dict, List, Optional, Tuple


def _compute_baseline_passive_learners_enabled() -> bool:
    """Read the fixed constants.py switch without exposing it to CLI users."""
    try:
        from constants import COMPUTE_BASELINE_PASSIVE_LEARNERS

        return bool(COMPUTE_BASELINE_PASSIVE_LEARNERS)
    except Exception:
        return True


# Passive helper functions live in a small standalone module.
# Do not import add_knowledge_mode_general_en.py here; that file contains the
# old heavy/offline analysis code and can be deleted without breaking runtime
# passive-gold analysis.
try:
    from passive_learners_core import (
        are_dfas_equivalent,
        infer_minimal_dfa_with_blue_fringe,
        infer_minimal_dfa_with_edsm,
        infer_minimal_dfa_with_rpni,
        normalize_word as _core_normalize_word,
    )
except Exception:
    are_dfas_equivalent = None
    infer_minimal_dfa_with_blue_fringe = None
    infer_minimal_dfa_with_edsm = None
    infer_minimal_dfa_with_rpni = None
    _core_normalize_word = None

INCLUDE_EQ_IN_KNOWLEDGE = True


def normalize_word(word: Any) -> str:
    if callable(_core_normalize_word):
        try:
            return _core_normalize_word(word)
        except Exception:
            pass
    if word is None:
        return ""
    s = html.unescape(str(word)).strip()
    if s in {"", "ε", "epsilon", "eps", "<eps>"}:
        return ""
    return "".join(s.split())


def _ensure_passive_helpers_loaded() -> None:
    # Kept for compatibility with the older code shape. Helpers are imported
    # above from passive_learners_core.py and never from add_knowledge mode.
    return None

def _algorithms() -> List[Tuple[str, str, Any]]:
    _ensure_passive_helpers_loaded()
    return [
        ("RPNI", "rpni", infer_minimal_dfa_with_rpni),
        ("EDSM", "edsm", infer_minimal_dfa_with_edsm),
        ("Blue-Fringe", "blue_fringe", infer_minimal_dfa_with_blue_fringe),
    ]

PASSIVE_CSV_COLUMNS = [
    "RPNI_FirstStep",
    "EDSM_FirstStep",
    "BlueFringe_FirstStep",
    "LStar_RPNI_FirstStep",
    "LStar_EDSM_FirstStep",
    "LStar_BlueFringe_FirstStep",
    "TTT_RPNI_FirstStep",
    "TTT_EDSM_FirstStep",
    "TTT_BlueFringe_FirstStep",
    "llm_gold_step",
    "lstar_gold_step",
    "ttt_gold_step",
    "llm_reached_gold_triangle",
    "lstar_reached_gold_triangle",
    "ttt_reached_gold_triangle",
    "llm_inefficient_steps",
    "lstar_inefficient_steps",
    "ttt_inefficient_steps",
]



def _bool_csv(value: bool) -> str:
    return "TRUE" if bool(value) else "FALSE"


def _value_or_minus_one(value: Any) -> str:
    return str(value if value is not None else -1)


def _html_value(value: Any) -> str:
    """HTML-only formatting: show -1/None as X, but do not affect CSV."""
    if value is None:
        return "X"
    s = str(value)
    return "X" if s == "-1" else s


def _snapshot_observations(accepted_words: set[str], rejected_words: set[str]) -> Dict[str, List[str]]:
    """Store exactly the membership observations sent to passive learners."""
    return {
        "accepted_words": sorted(accepted_words),
        "rejected_words": sorted(rejected_words),
    }


def _format_word_for_html(word: Any) -> str:
    w = "" if word is None else str(word)
    return "ε" if w == "" else w


def _render_observation_list_html(title: str, words: Any) -> str:
    if not isinstance(words, list):
        words = []
    if not words:
        body = '<span class="passive_obs_empty">none</span>'
    else:
        body = "".join(f"<code>{html.escape(_format_word_for_html(w))}</code>" for w in words)
    return (
        '<div class="passive_obs_group">'
        f'<div class="passive_obs_title">{html.escape(title)} ({len(words)})</div>'
        f'<div class="passive_obs_words">{body}</div>'
        '</div>'
    )


def _inefficient_steps(total_queries: Any, gold_step: Optional[int], reached_gold: bool) -> str:
    if not reached_gold or gold_step is None:
        return "-1"
    try:
        return str(max(0, int(total_queries) - int(gold_step)))
    except Exception:
        return "-1"


def _empty_stats() -> Dict[str, Any]:
    return {
        "RPNI_FirstStep": "-1",
        "EDSM_FirstStep": "-1",
        "BlueFringe_FirstStep": "-1",
        "LStar_RPNI_FirstStep": "-1",
        "LStar_EDSM_FirstStep": "-1",
        "LStar_BlueFringe_FirstStep": "-1",
        "TTT_RPNI_FirstStep": "-1",
        "TTT_EDSM_FirstStep": "-1",
        "TTT_BlueFringe_FirstStep": "-1",
        "llm_gold_step": "-1",
        "lstar_gold_step": "-1",
        "ttt_gold_step": "-1",
        "llm_reached_gold_triangle": "FALSE",
        "lstar_reached_gold_triangle": "FALSE",
        "ttt_reached_gold_triangle": "FALSE",
        "llm_inefficient_steps": "-1",
        "lstar_inefficient_steps": "-1",
        "ttt_inefficient_steps": "-1",
    }


def _evaluate_one_passive(
    algorithm_name: str,
    infer_fn: Any,
    target_dfa: Any,
    accepted_words: set[str],
    rejected_words: set[str],
) -> Tuple[Optional[bool], str]:
    _ensure_passive_helpers_loaded()
    if infer_fn is None or are_dfas_equivalent is None:
        return None, f"{algorithm_name} unavailable"
    try:
        learned = infer_fn(target_dfa, {w: True for w in accepted_words}, {w: False for w in rejected_words})
        equivalent, message = are_dfas_equivalent(learned, target_dfa)
        return equivalent, message
    except Exception as exc:
        return None, f"{algorithm_name} failed: {exc}"


def _normalize_strategy_name(name: Any) -> Optional[str]:
    s = str(name).upper().replace(" ", "")
    if "LSTAR" in s or "L*" in s:
        return "lstar"
    if "TTT" in s:
        return "ttt"
    if "LLM" in s:
        return "llm"
    return None


def _add_knowledge(
    *,
    strategy_key: str,
    is_eq: bool,
    word: Optional[str],
    accepted: Optional[bool],
    accepted_words: set[str],
    rejected_words: set[str],
) -> None:
    if word is None or accepted is None:
        return
    if is_eq and strategy_key in {"lstar", "ttt"}:
        return
    if is_eq and strategy_key == "llm" and not INCLUDE_EQ_IN_KNOWLEDGE:
        return
    w = normalize_word(word)
    if accepted:
        accepted_words.add(w)
        rejected_words.discard(w)
    else:
        rejected_words.add(w)
        accepted_words.discard(w)


def _event_from_tool_output(out: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = out.get("output") or {}
    if not isinstance(payload, dict):
        payload = {}
    tool_name = str(out.get("tool_name", "")).strip().replace("-", "_").lower()
    try:
        step = int(out.get("call_count", 0))
    except Exception:
        step = 0

    if tool_name in {"is_word_in_language", "membership_query", "mq"}:
        return {
            "step": step,
            "is_eq": False,
            "word": payload.get("word"),
            "accepted": payload.get("accepted"),
        }

    if tool_name in {"evaluate_dfa_candidate", "equivalence_query", "eq"}:
        witness = payload.get("witness_word")
        accepted = None
        ks = out.get("knowledge_state")
        if isinstance(ks, dict) and witness:
            acc = {normalize_word(x) for x in (ks.get("words_accepted_by_dfa") or [])}
            rej = {normalize_word(x) for x in (ks.get("words_rejected_by_dfa") or [])}
            w = normalize_word(witness)
            if w in acc:
                accepted = True
            elif w in rej:
                accepted = False
        return {
            "step": step,
            "is_eq": True,
            "word": witness,
            "accepted": accepted,
        }

    return None


def _strategy_events(result: Any) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for idx, item in enumerate(getattr(result, "history", []) or [], start=1):
        if not isinstance(item, (tuple, list)) or not item:
            continue
        kind = str(item[0]).strip().replace("-", "_").lower()
        if kind == "mq":
            events.append({
                "step": idx,
                "is_eq": False,
                "word": item[1] if len(item) > 1 else None,
                "accepted": item[2] if len(item) > 2 else None,
            })
        elif kind == "eq":
            events.append({
                "step": idx,
                "is_eq": True,
                "word": item[1] if len(item) > 1 else None,
                "accepted": None,
            })
    return events


def _new_runtime_state() -> Dict[str, Any]:
    return {
        "accepted_words": set(),
        "rejected_words": set(),
        "first": {"rpni": None, "edsm": None, "blue_fringe": None},
        "steps": [],
        "lock": threading.Lock(),
        "queue": queue.Queue(),
        "worker": None,
        "worker_errors": [],
    }


def initialize_runtime_passive_gold(owner: Any) -> None:
    """Prepare the incremental LLM passive state. Does not run passive learners."""
    owner._passive_gold_llm_state = _new_runtime_state()
    owner._passive_gold_baseline_analysis = None
    owner._passive_gold_analysis_cache = None


def _evaluate_llm_snapshot(owner: Any, step: int, accepted_words: set[str], rejected_words: set[str]) -> None:
    """Evaluate passive learners on a snapshot that was true BEFORE this step.

    This is intentionally not allowed to include the observation returned at
    the same tool call. That observation is added to the state only after the
    snapshot has been queued, so it can affect later steps only.
    """
    target_dfa = getattr(getattr(owner, "game", None), "dfa", None)
    state = getattr(owner, "_passive_gold_llm_state", None)
    if target_dfa is None or not isinstance(state, dict):
        return

    passive_results: Dict[str, Dict[str, Any]] = {}

    with state["lock"]:
        first_before = dict(state["first"])

    for display_name, key, infer_fn in _algorithms():
        if first_before.get(key) is not None:
            passive_results[key] = {
                "algorithm": display_name,
                "equivalent": True,
                "message": "Already succeeded earlier; skipped on this step.",
            }
            continue
        equivalent, message = _evaluate_one_passive(display_name, infer_fn, target_dfa, accepted_words, rejected_words)
        passive_results[key] = {"algorithm": display_name, "equivalent": equivalent, "message": message}

    observations = _snapshot_observations(accepted_words, rejected_words)

    with state["lock"]:
        first = state["first"]
        for _display_name, key, _infer_fn in _algorithms():
            if passive_results.get(key, {}).get("equivalent") is True and first.get(key) is None:
                first[key] = step
        state["steps"].append({
            "strategy": "llm",
            "step": step,
            "accepted_count": len(accepted_words),
            "rejected_count": len(rejected_words),
            "accepted_words": observations["accepted_words"],
            "rejected_words": observations["rejected_words"],
            "passive_results": passive_results,
        })
    owner._passive_gold_analysis_cache = None


def _passive_worker(owner: Any) -> None:
    state = getattr(owner, "_passive_gold_llm_state", None)
    if not isinstance(state, dict):
        return
    q = state.get("queue")
    if not isinstance(q, queue.Queue):
        return
    while True:
        item = q.get()
        try:
            if item is None:
                return
            step, accepted_words, rejected_words = item
            _evaluate_llm_snapshot(owner, int(step), set(accepted_words), set(rejected_words))
        except Exception as exc:
            try:
                state["worker_errors"].append(str(exc))
            except Exception:
                pass
        finally:
            q.task_done()


def _ensure_llm_worker(owner: Any) -> None:
    state = getattr(owner, "_passive_gold_llm_state", None)
    if not isinstance(state, dict):
        initialize_runtime_passive_gold(owner)
        state = owner._passive_gold_llm_state

    worker = state.get("worker")
    if isinstance(worker, threading.Thread) and worker.is_alive():
        return

    worker = threading.Thread(target=_passive_worker, args=(owner,), daemon=True, name="passive-gold-llm")
    state["worker"] = worker
    worker.start()


def finalize_runtime_passive_gold(owner: Any) -> None:
    """Wait for queued LLM passive computations before exporting results."""
    state = getattr(owner, "_passive_gold_llm_state", None)
    if not isinstance(state, dict):
        return
    q = state.get("queue")
    if isinstance(q, queue.Queue):
        q.join()

def update_runtime_passive_gold_from_tool_reply(owner: Any, tool_reply: Dict[str, Any]) -> None:
    """Queue LLM passive analysis without blocking the game.

    For each tool call, passive learners receive only the observations known
    BEFORE that tool call's returned observation. The new observation is added
    afterwards and can influence only following tool calls.
    """
    if not isinstance(getattr(owner, "_passive_gold_llm_state", None), dict):
        initialize_runtime_passive_gold(owner)

    outs = tool_reply.get("tool_outputs", []) if isinstance(tool_reply, dict) else []
    if not isinstance(outs, list):
        return

    _ensure_llm_worker(owner)
    state = owner._passive_gold_llm_state
    q = state.get("queue")
    if not isinstance(q, queue.Queue):
        return

    for out in outs:
        if not isinstance(out, dict):
            continue
        event = _event_from_tool_output(out)
        if not event:
            continue
        step = int(event.get("step") or getattr(owner, "_call_counter", 0) or 0)

        # Snapshot BEFORE adding the current observation. This fixes the issue
        # where the observation returned at step N was incorrectly included in
        # the passive learner input for step N.
        #
        # Also: once a passive learner already succeeded for the LLM path, it
        # is not evaluated again on later steps. If all passive learners already
        # succeeded, do not enqueue any more LLM passive work at all. The new
        # observation is still stored below for completeness/export.
        with state["lock"]:
            accepted_snapshot = set(state["accepted_words"])
            rejected_snapshot = set(state["rejected_words"])
            first_snapshot = dict(state.get("first") or {})

        if not all(first_snapshot.get(k) is not None for k in ("rpni", "edsm", "blue_fringe")):
            q.put((step, accepted_snapshot, rejected_snapshot))

        # Add the current observation only after the snapshot was queued/skipped.
        with state["lock"]:
            _add_knowledge(
                strategy_key="llm",
                is_eq=bool(event.get("is_eq")),
                word=event.get("word"),
                accepted=event.get("accepted"),
                accepted_words=state["accepted_words"],
                rejected_words=state["rejected_words"],
            )

        owner._passive_gold_analysis_cache = None


def _passive_ui_snapshot(owner: Any, call_count: Any) -> Dict[str, Any]:
    """Build a compact before-this-tool-call passive-learning snapshot for launcher.py.

    The launcher prints this snapshot before update_runtime_passive_gold_from_tool_reply()
    adds the current oracle answer. Therefore call #1 shows empty dictionaries,
    and each later card shows the observations available before that query.
    """
    target_dfa = getattr(getattr(owner, "game", None), "dfa", None)
    state = getattr(owner, "_passive_gold_llm_state", None)
    if target_dfa is None or not isinstance(state, dict):
        return {
            "call": str(call_count or ""),
            "accepted_words": [],
            "rejected_words": [],
            "results": [],
            "error": "Passive learner state is not available.",
        }

    try:
        with state["lock"]:
            accepted_words = set(state.get("accepted_words") or set())
            rejected_words = set(state.get("rejected_words") or set())
    except Exception:
        accepted_words = set(state.get("accepted_words") or set())
        rejected_words = set(state.get("rejected_words") or set())

    results: List[Dict[str, Any]] = []
    for display_name, key, infer_fn in _algorithms():
        equivalent, message = _evaluate_one_passive(display_name, infer_fn, target_dfa, accepted_words, rejected_words)
        results.append({
            "key": key,
            "algorithm": display_name,
            "success": bool(equivalent is True),
            "status": "success" if equivalent is True else "failure" if equivalent is False else "unavailable",
            "message": str(message or ""),
        })

    observations = _snapshot_observations(accepted_words, rejected_words)
    return {
        "call": str(call_count or ""),
        "accepted_words": observations["accepted_words"],
        "rejected_words": observations["rejected_words"],
        "results": results,
    }


def print_passive_learning_ui_snapshot(owner: Any, tool_reply: Dict[str, Any]) -> None:
    """Print one parseable launcher line per accepted tool call."""
    outs = tool_reply.get("tool_outputs", []) if isinstance(tool_reply, dict) else []
    if not isinstance(outs, list):
        return
    for out in outs:
        if not isinstance(out, dict):
            continue
        call_count = out.get("call_count", "")
        snapshot = _passive_ui_snapshot(owner, call_count)
        print(
            "PASSIVE_LEARNING_ANALYSIS::CALL="
            + str(call_count or "")
            + "::JSON="
            + json.dumps(snapshot, ensure_ascii=False, sort_keys=True),
            flush=True,
        )


def _compute_for_strategy(
    *,
    strategy_key: str,
    events: List[Dict[str, Any]],
    target_dfa: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    accepted_words: set[str] = set()
    rejected_words: set[str] = set()
    first: Dict[str, Optional[int]] = {"rpni": None, "edsm": None, "blue_fringe": None}
    step_rows: List[Dict[str, Any]] = []

    for event in events:
        step = int(event.get("step", len(step_rows) + 1))

        # Evaluate using observations known before this event, not including
        # the observation returned by this same step.
        passive_results: Dict[str, Dict[str, Any]] = {}
        for display_name, key, infer_fn in _algorithms():
            if first[key] is not None:
                passive_results[key] = {"algorithm": display_name, "equivalent": True, "message": "Already succeeded earlier; skipped on this step."}
                continue
            equivalent, message = _evaluate_one_passive(display_name, infer_fn, target_dfa, accepted_words, rejected_words)
            passive_results[key] = {"algorithm": display_name, "equivalent": equivalent, "message": message}
            if equivalent is True and first[key] is None:
                first[key] = step

        observations = _snapshot_observations(accepted_words, rejected_words)
        step_rows.append({
            "strategy": strategy_key,
            "step": step,
            "accepted_count": len(accepted_words),
            "rejected_count": len(rejected_words),
            "accepted_words": observations["accepted_words"],
            "rejected_words": observations["rejected_words"],
            "passive_results": passive_results,
        })

        # Make this step's observation available only for later steps.
        _add_knowledge(
            strategy_key=strategy_key,
            is_eq=bool(event.get("is_eq")),
            word=event.get("word"),
            accepted=event.get("accepted"),
            accepted_words=accepted_words,
            rejected_words=rejected_words,
        )

    gold_candidates = [v for v in first.values() if v is not None]
    return {
        "rpni_first_step": first["rpni"],
        "edsm_first_step": first["edsm"],
        "blue_fringe_first_step": first["blue_fringe"],
        "gold_step": min(gold_candidates) if gold_candidates else None,
        "reached_gold": bool(gold_candidates),
    }, step_rows


def _llm_result_from_runtime(owner: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    state = getattr(owner, "_passive_gold_llm_state", None)
    if not isinstance(state, dict):
        return {
            "rpni_first_step": None,
            "edsm_first_step": None,
            "blue_fringe_first_step": None,
            "gold_step": None,
            "reached_gold": False,
        }, []
    try:
        with state["lock"]:
            first = dict(state.get("first") or {})
            steps = list(state.get("steps") or [])
    except Exception:
        first = dict(state.get("first") or {})
        steps = list(state.get("steps") or [])
    gold_candidates = [v for v in first.values() if v is not None]
    return {
        "rpni_first_step": first.get("rpni"),
        "edsm_first_step": first.get("edsm"),
        "blue_fringe_first_step": first.get("blue_fringe"),
        "gold_step": min(gold_candidates) if gold_candidates else None,
        "reached_gold": bool(gold_candidates),
    }, steps


def _compute_baselines(owner: Any) -> Dict[str, Any]:
    cached = getattr(owner, "_passive_gold_baseline_analysis", None)
    if isinstance(cached, dict):
        return cached

    stats: Dict[str, Any] = {}
    steps_by_strategy: Dict[str, List[Dict[str, Any]]] = {"lstar": [], "ttt": []}

    # LLM passive analysis is updated online during the game. Baseline passive
    # analysis for L* and TTT can be expensive, so it is controlled by a fixed
    # constant in constants.py and is not user-configurable from the CLI.
    if not _compute_baseline_passive_learners_enabled():
        out = {"stats": stats, "steps_by_strategy": steps_by_strategy}
        owner._passive_gold_baseline_analysis = out
        return out

    target_dfa = getattr(getattr(owner, "game", None), "dfa", None)
    totals = {"lstar": 0, "ttt": 0}

    if target_dfa is not None:
        runs = getattr(target_dfa, "strategy_results", None)
        latest = runs[-1] if isinstance(runs, list) and runs else None
        if isinstance(latest, dict):
            for strat_name, res in latest.items():
                key = _normalize_strategy_name(strat_name)
                if key not in {"lstar", "ttt"}:
                    continue
                totals[key] = getattr(res, "total_queries", 0)
                result, step_rows = _compute_for_strategy(strategy_key=key, events=_strategy_events(res), target_dfa=target_dfa)
                steps_by_strategy[key] = step_rows
                prefix = "LStar" if key == "lstar" else "TTT"
                stats[f"{prefix}_RPNI_FirstStep"] = _value_or_minus_one(result["rpni_first_step"])
                stats[f"{prefix}_EDSM_FirstStep"] = _value_or_minus_one(result["edsm_first_step"])
                stats[f"{prefix}_BlueFringe_FirstStep"] = _value_or_minus_one(result["blue_fringe_first_step"])
                stats[f"{key}_gold_step"] = _value_or_minus_one(result["gold_step"])
                stats[f"{key}_reached_gold_triangle"] = _bool_csv(result["reached_gold"])
                stats[f"{key}_inefficient_steps"] = _inefficient_steps(
                    totals.get(key, 0), result["gold_step"], result["reached_gold"]
                )

    out = {"stats": stats, "steps_by_strategy": steps_by_strategy}
    owner._passive_gold_baseline_analysis = out
    return out


def get_passive_gold_analysis(owner: Any, *, include_baselines: bool = True) -> Dict[str, Any]:
    # Export/HTML happen after the game. At that point wait for any queued
    # background LLM passive computations, but never block the game loop itself.
    finalize_runtime_passive_gold(owner)
    cached = getattr(owner, "_passive_gold_analysis_cache", None)
    if isinstance(cached, dict) and include_baselines:
        return cached

    stats = _empty_stats()
    steps_by_strategy: Dict[str, List[Dict[str, Any]]] = {"llm": [], "lstar": [], "ttt": []}

    llm_result, llm_steps = _llm_result_from_runtime(owner)
    steps_by_strategy["llm"] = llm_steps
    stats["RPNI_FirstStep"] = _value_or_minus_one(llm_result["rpni_first_step"])
    stats["EDSM_FirstStep"] = _value_or_minus_one(llm_result["edsm_first_step"])
    stats["BlueFringe_FirstStep"] = _value_or_minus_one(llm_result["blue_fringe_first_step"])
    stats["llm_gold_step"] = _value_or_minus_one(llm_result["gold_step"])
    stats["llm_reached_gold_triangle"] = _bool_csv(llm_result["reached_gold"])
    stats["llm_inefficient_steps"] = _inefficient_steps(
        getattr(owner, "_compute_llm_total_queries", lambda: getattr(owner, "_call_counter", 0))(),
        llm_result["gold_step"],
        llm_result["reached_gold"],
    )

    if include_baselines and _compute_baseline_passive_learners_enabled():
        baseline = _compute_baselines(owner)
        stats.update(baseline.get("stats", {}))
        steps_by_strategy.update(baseline.get("steps_by_strategy", {}))
        out = {"stats": stats, "steps_by_strategy": steps_by_strategy}
        owner._passive_gold_analysis_cache = out
        return out

    return {"stats": stats, "steps_by_strategy": steps_by_strategy}


# Backwards-compatible name used by older code paths.
def compute_passive_gold_analysis(owner: Any) -> Dict[str, Any]:
    return get_passive_gold_analysis(owner, include_baselines=True)


def passive_columns_for_csv(owner: Any) -> Dict[str, str]:
    analysis = get_passive_gold_analysis(owner, include_baselines=True)
    stats = analysis.get("stats", {})
    return {col: str(stats.get(col, "-1" if not col.endswith("reached_gold_triangle") else "FALSE")) for col in PASSIVE_CSV_COLUMNS}


def render_passive_summary_html(analysis: Dict[str, Any], strategy_key: str) -> str:
    stats = analysis.get("stats", {}) if isinstance(analysis, dict) else {}
    if strategy_key == "llm":
        items = [
            ("RPNI_FirstStep", stats.get("RPNI_FirstStep", "-1")),
            ("EDSM_FirstStep", stats.get("EDSM_FirstStep", "-1")),
            ("BlueFringe_FirstStep", stats.get("BlueFringe_FirstStep", "-1")),
            ("llm_gold_step (minimum of passive algorithms)", stats.get("llm_gold_step", "-1")),
            ("llm_inefficient_steps", stats.get("llm_inefficient_steps", "-1")),
        ]
    else:
        prefix = "LStar" if strategy_key == "lstar" else "TTT"
        items = [
            (f"{prefix}_RPNI_FirstStep", stats.get(f"{prefix}_RPNI_FirstStep", "-1")),
            (f"{prefix}_EDSM_FirstStep", stats.get(f"{prefix}_EDSM_FirstStep", "-1")),
            (f"{prefix}_BlueFringe_FirstStep", stats.get(f"{prefix}_BlueFringe_FirstStep", "-1")),
            (f"{strategy_key}_gold_step", stats.get(f"{strategy_key}_gold_step", "-1")),
            (f"{strategy_key}_inefficient_steps", stats.get(f"{strategy_key}_inefficient_steps", "-1")),
        ]
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(_html_value(v))}</td></tr>" for k, v in items)
    return f"""
    <details class="payload passive_gold_summary" open>
      <summary>Passive Learners</summary>
      <table class="passive_gold_table">{rows}</table>
    </details>
    """


def render_passive_step_html(analysis: Dict[str, Any], strategy_key: str, step: Any) -> str:
    try:
        step_int = int(step)
    except Exception:
        return ""
    rows = ((analysis or {}).get("steps_by_strategy") or {}).get(strategy_key, [])
    row = next((r for r in rows if int(r.get("step", -1)) == step_int), None)
    if not row:
        return ""
    parts: List[str] = []
    for key in ["rpni", "edsm", "blue_fringe"]:
        item = (row.get("passive_results") or {}).get(key) or {}
        algorithm = str(item.get("algorithm", key))
        eq = item.get("equivalent")
        if eq is True:
            status = "equivalent to target"
            cls = "passive_ok"
        elif eq is False:
            status = "not equivalent"
            cls = "passive_bad"
        else:
            status = "not available / not enough information"
            cls = "passive_unknown"
        msg = str(item.get("message", ""))
        observations_html = (
            _render_observation_list_html("Accepted observations", row.get("accepted_words") or [])
            + _render_observation_list_html("Rejected observations", row.get("rejected_words") or [])
        )
        parts.append(
            f"""
            <details class="payload passive_learning_payload {html.escape(key)}_payload">
              <summary>{html.escape(algorithm)}</summary>
              <div class="passive_box {cls}">
                <div><b>{html.escape(status)}</b></div>
                <div>observations sent to this learner before this step: accepted={html.escape(str(row.get('accepted_count', 0)))}, rejected={html.escape(str(row.get('rejected_count', 0)))}</div>
                <div class="passive_observations">{observations_html}</div>
                <pre>{html.escape(msg)}</pre>
              </div>
            </details>
            """
        )
    return "\n".join(parts)
