"""
Runtime analysis of LLM hypothesis behavior.

Tracks two binary diagnostics during the interactive game:
1. Whether the sequence of submitted EQ hypotheses violates strict monotonicity
   in the number of states.
2. Whether the number of EQ hypotheses submitted by the LLM is strictly larger
   than the number of states in the target DFA.

CSV convention: 1 means the property/violation occurred, 0 otherwise.
"""
from __future__ import annotations

import html
from typing import Any, Dict, List, Optional


HYPOTHESIS_CSV_COLUMNS = [
    "llm_hypothesis_monotonicity_broken",
    "llm_eq_count_gt_target_states",
]


def _state_count(dfa: Any) -> Optional[int]:
    if dfa is None:
        return None
    states = getattr(dfa, "states", None)
    if states is None:
        return None
    try:
        return int(len(states))
    except Exception:
        return None


def _target_state_count(owner: Any) -> Optional[int]:
    return _state_count(getattr(getattr(owner, "game", None), "dfa", None))


def initialize_runtime_hypothesis_analysis(owner: Any) -> None:
    owner._hypothesis_runtime_state = {
        "items": [],
        "max_states_so_far": None,
        "monotonicity_broken": 0,
        "eq_count_gt_target_states": 0,
    }
    owner._hypothesis_analysis_cache = None


def update_runtime_hypothesis_analysis_from_candidate(owner: Any, *, step: int, hypothesis_dfa: Any) -> None:
    """Update monotonicity and EQ-count diagnostics immediately after an EQ."""
    state = getattr(owner, "_hypothesis_runtime_state", None)
    if not isinstance(state, dict):
        initialize_runtime_hypothesis_analysis(owner)
        state = owner._hypothesis_runtime_state

    n_states = _state_count(hypothesis_dfa)
    previous_max = state.get("max_states_so_far")

    # Strict monotonicity means every new hypothesis must have more states than
    # every previous hypothesis. Therefore <= previous max breaks monotonicity.
    broke_here = 0
    if n_states is not None and previous_max is not None and n_states <= int(previous_max):
        broke_here = 1
        state["monotonicity_broken"] = 1

    if n_states is not None:
        if previous_max is None or n_states > int(previous_max):
            state["max_states_so_far"] = n_states

    item = {
        "step": int(step),
        "hypothesis_state_count": n_states,
        "previous_max_state_count": previous_max,
        "monotonicity_broken_here": broke_here,
    }
    state.setdefault("items", []).append(item)

    target_n = _target_state_count(owner)
    if target_n is not None and len(state.get("items", [])) > int(target_n):
        state["eq_count_gt_target_states"] = 1

    owner._hypothesis_analysis_cache = None


def get_hypothesis_runtime_analysis(owner: Any) -> Dict[str, Any]:
    cached = getattr(owner, "_hypothesis_analysis_cache", None)
    if isinstance(cached, dict):
        return cached

    state = getattr(owner, "_hypothesis_runtime_state", None)
    if not isinstance(state, dict):
        # Fallback for old runs: reconstruct from eq_dfa_guesses at export time.
        initialize_runtime_hypothesis_analysis(owner)
        for item in getattr(owner, "eq_dfa_guesses", []) or []:
            try:
                step, cand, _witness = item
            except Exception:
                continue
            update_runtime_hypothesis_analysis_from_candidate(owner, step=int(step), hypothesis_dfa=cand)
        state = owner._hypothesis_runtime_state

    target_n = _target_state_count(owner)
    items = list(state.get("items", []) or [])
    out = {
        "target_state_count": target_n,
        "llm_hypothesis_count": len(items),
        "llm_hypothesis_monotonicity_broken": int(bool(state.get("monotonicity_broken", 0))),
        "llm_eq_count_gt_target_states": int(bool(state.get("eq_count_gt_target_states", 0))),
        "items": items,
    }
    owner._hypothesis_analysis_cache = out
    return out


def hypothesis_columns_for_csv(owner: Any) -> Dict[str, str]:
    analysis = get_hypothesis_runtime_analysis(owner)
    return {
        "llm_hypothesis_monotonicity_broken": str(int(analysis.get("llm_hypothesis_monotonicity_broken", 0))),
        "llm_eq_count_gt_target_states": str(int(analysis.get("llm_eq_count_gt_target_states", 0))),
    }


def render_hypothesis_runtime_html(owner: Any) -> str:
    analysis = get_hypothesis_runtime_analysis(owner)
    rows = [
        ("LLM hypothesis monotonicity broken", analysis.get("llm_hypothesis_monotonicity_broken", 0)),
        ("LLM #hypotheses > target #states", analysis.get("llm_eq_count_gt_target_states", 0)),
        ("LLM hypothesis count", analysis.get("llm_hypothesis_count", 0)),
        ("Target state count", analysis.get("target_state_count", "X")),
    ]
    body = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v if v is not None else 'X'))}</td></tr>"
        for k, v in rows
    )
    return f"""
    <details class="payload hypothesis_runtime_summary">
    <summary>Hypothesis Diagnostics</summary>
    <table class="passive_gold_table">{body}</table>
    </details>
    """
