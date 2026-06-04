"""
Runtime symmetric-difference similarity analysis.

This module is independent of add_knowledge_mode_general_en.py and of the old
post-processing script. It computes, during export, the similarity between each
EQ hypothesis and the target DFA:

    similarity = 1 - |L(h) Δ L(target)| / |Sigma^{<=k}|

Only the symmetric-difference metric is supported.
"""
from __future__ import annotations

import ast
import html
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from output_paths import get_artifact_dir
from html_code.llm_comparison_html import read_if_path

getcontext().prec = 80

LLM_SIMILARITY_COLUMN = "llm_symdiff_similarity_by_step"
LSTAR_SIMILARITY_COLUMN = "lstar_symdiff_similarity_by_step"
TTT_SIMILARITY_COLUMN = "ttt_symdiff_similarity_by_step"
LANGUAGE_SIMILARITY_CSV_COLUMNS = [
    LLM_SIMILARITY_COLUMN,
    LSTAR_SIMILARITY_COLUMN,
    TTT_SIMILARITY_COLUMN,
]


@dataclass
class SimpleDFA:
    states: set
    input_symbols: set
    transitions: dict
    initial_state: Any
    final_states: set


def _compute_baselines_enabled() -> bool:
    try:
        from constants import COMPUTE_LANGUAGE_SIMILARITY_FOR_BASELINES
        return bool(COMPUTE_LANGUAGE_SIMILARITY_FOR_BASELINES)
    except Exception:
        return True


def _language_similarity_k() -> int:
    try:
        from constants import LANGUAGE_SIMILARITY_MAX_WORD_LENGTH
        return int(LANGUAGE_SIMILARITY_MAX_WORD_LENGTH)
    except Exception:
        return 200


def _as_file_url(path: str) -> str:
    try:
        return Path(os.path.abspath(path)).as_uri()
    except Exception:
        return os.path.abspath(path)


def _format_decimal_short(value: Decimal) -> str:
    s = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _format_decimal_csv(value: Decimal) -> str:
    return str(value)


def _extract_arg(text: str, name: str) -> str:
    marker = f"{name}="
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"missing {name}")
    i = start + len(marker)
    depth = 0
    in_str: Optional[str] = None
    escape = False
    out = []
    while i < len(text):
        ch = text[i]
        if in_str:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in {"'", '"'}:
            in_str = ch
            out.append(ch)
            i += 1
            continue
        if ch in "([{":
            depth += 1
            out.append(ch)
            i += 1
            continue
        if ch in ")]}":
            if depth > 0:
                depth -= 1
                out.append(ch)
                i += 1
                continue
            break
        if ch == "," and depth == 0:
            break
        if ch == ")" and depth == 0:
            break
        out.append(ch)
        i += 1
    return "".join(out).strip()


def _literal_arg(text: str, name: str) -> Any:
    return ast.literal_eval(_extract_arg(text, name))


def _simple_dfa_from_text(text: str) -> SimpleDFA:
    s = str(text).strip()
    states = set(_literal_arg(s, "states"))
    input_symbols = set(_literal_arg(s, "input_symbols"))
    transitions = _literal_arg(s, "transitions")
    initial_state = _literal_arg(s, "initial_state")
    final_states = set(_literal_arg(s, "final_states"))
    return SimpleDFA(
        states=states,
        input_symbols=input_symbols,
        transitions=transitions,
        initial_state=initial_state,
        final_states=final_states,
    )


def _simple_dfa_from_obj(obj: Any) -> SimpleDFA:
    if isinstance(obj, SimpleDFA):
        return obj
    if isinstance(obj, str):
        return _simple_dfa_from_text(obj)
    return SimpleDFA(
        states=set(getattr(obj, "states")),
        input_symbols=set(getattr(obj, "input_symbols")),
        transitions=dict(getattr(obj, "transitions")),
        initial_state=getattr(obj, "initial_state"),
        final_states=set(getattr(obj, "final_states")),
    )


def _transition_with_sink(dfa: SimpleDFA, state: Any, symbol: Any, sink_state: str) -> Any:
    if state == sink_state:
        return sink_state
    return dfa.transitions.get(state, {}).get(symbol, sink_state)


def _is_final_with_sink(dfa: SimpleDFA, state: Any, sink_state: str) -> bool:
    if state == sink_state:
        return False
    return state in dfa.final_states


def compute_symdiff_similarity_stats(target_dfa: Any, hypothesis_dfa: Any, k: Optional[int] = None) -> Dict[str, Any]:
    """Return symmetric-difference distance and similarity up to length k."""
    k = _language_similarity_k() if k is None else int(k)
    dfa1 = _simple_dfa_from_obj(target_dfa)
    dfa2 = _simple_dfa_from_obj(hypothesis_dfa)

    alphabet = sorted(set(dfa1.input_symbols) | set(dfa2.input_symbols), key=str)
    sink1 = "__SINK_TARGET__"
    sink2 = "__SINK_HYPOTHESIS__"

    current: Dict[Tuple[Any, Any], int] = {(dfa1.initial_state, dfa2.initial_state): 1}
    total_diff_count = 0
    exact_counts: List[int] = []

    for length in range(k + 1):
        exact_diff_count = 0
        for (s1, s2), count in current.items():
            accept1 = _is_final_with_sink(dfa1, s1, sink1)
            accept2 = _is_final_with_sink(dfa2, s2, sink2)
            if accept1 != accept2:
                exact_diff_count += count
        exact_counts.append(exact_diff_count)
        total_diff_count += exact_diff_count

        if length == k:
            break

        next_current: Dict[Tuple[Any, Any], int] = {}
        for (s1, s2), count in current.items():
            for symbol in alphabet:
                t1 = _transition_with_sink(dfa1, s1, symbol, sink1)
                t2 = _transition_with_sink(dfa2, s2, symbol, sink2)
                pair = (t1, t2)
                next_current[pair] = next_current.get(pair, 0) + count
        current = next_current

    total_words_up_to_k = sum(len(alphabet) ** length for length in range(k + 1))
    distance_decimal = (
        Decimal(total_diff_count) / Decimal(total_words_up_to_k)
        if total_words_up_to_k > 0
        else Decimal(0)
    )
    similarity_decimal = Decimal(1) - distance_decimal

    return {
        "k": k,
        "metric": "symdiff",
        "alphabet": alphabet,
        "alphabet_size": len(alphabet),
        "total_words_up_to_k": total_words_up_to_k,
        "symmetric_difference_count": total_diff_count,
        "distance_ratio_decimal": distance_decimal,
        "similarity_decimal": similarity_decimal,
        "similarity_float": float(similarity_decimal),
        "exact_counts_by_length": exact_counts,
    }


def _draw_or_text(dfa: Any) -> str:
    try:
        if hasattr(dfa, "draw"):
            return read_if_path(dfa.draw())
    except Exception:
        pass
    return f"<pre style='white-space:pre-wrap;font-family:monospace;padding:16px'>{html.escape(str(dfa))}</pre>"


def _write_similarity_detail_html(
    *,
    strategy: str,
    step: int,
    target_dfa: Any,
    hypothesis_dfa: Any,
    stats: Dict[str, Any],
) -> str:
    out_dir = get_artifact_dir("language_similarity_details")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{strategy}_step_{step}_symdiff_similarity_k_{stats['k']}_{ts}.html"
    out_path = out_dir / filename

    left_html = _draw_or_text(target_dfa)
    right_html = _draw_or_text(hypothesis_dfa)
    left_srcdoc = html.escape(left_html, quote=True)
    right_srcdoc = html.escape(right_html, quote=True)

    rows = "".join(
        f"<tr><td>{length}</td><td>{count}</td></tr>"
        for length, count in enumerate(stats["exact_counts_by_length"])
    )

    page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Symmetric Difference Similarity</title>
<style>
html, body {{ height:100%; margin:0; font-family:Arial, sans-serif; }}
.container {{ display:flex; height:100vh; }}
.side {{ flex:1; min-width:0; }}
.center {{ width:430px; padding:16px; box-sizing:border-box; border-left:1px solid #ddd; border-right:1px solid #ddd; overflow:auto; }}
iframe {{ width:100%; height:100%; border:0; }}
h2 {{ margin:0 0 12px 0; }}
.metric {{ font-size:18px; font-weight:700; margin:12px 0; }}
.small {{ color:#555; margin:6px 0; font-size:13px; }}
table {{ border-collapse:collapse; width:100%; margin-top:14px; }}
th,td {{ border:1px solid #ddd; padding:5px 7px; text-align:left; }}
th {{ background:#f6f6f6; }}
</style>
</head>
<body>
<div class="container">
  <div class="side"><iframe srcdoc="{left_srcdoc}"></iframe></div>
  <div class="center">
    <h2>Symmetric Difference Similarity</h2>
    <div class="small"><b>Strategy:</b> {html.escape(strategy)}</div>
    <div class="small"><b>Step:</b> {step}</div>
    <div class="small"><b>k:</b> {stats['k']}</div>
    <div class="small"><b>Metric:</b> 1 - symdiff distance</div>
    <div class="metric">sim = {_format_decimal_short(stats['similarity_decimal'])}</div>
    <div class="small"><b>Full similarity:</b> {stats['similarity_decimal']}</div>
    <div class="small"><b>Distance:</b> {stats['distance_ratio_decimal']}</div>
    <div class="small"><b>Different words:</b> {stats['symmetric_difference_count']}</div>
    <div class="small"><b>Total words up to k:</b> {stats['total_words_up_to_k']}</div>
    <div class="small"><b>Alphabet:</b> {html.escape(str(stats['alphabet']))}</div>
    <div class="small">Left: target DFA</div>
    <div class="small">Right: hypothesis DFA</div>
    <table><thead><tr><th>Length</th><th># words in symmetric difference</th></tr></thead><tbody>{rows}</tbody></table>
  </div>
  <div class="side"><iframe srcdoc="{right_srcdoc}"></iframe></div>
</div>
</body>
</html>"""
    out_path.write_text(page, encoding="utf-8")
    return str(out_path)


def _strategy_display(strategy: str) -> str:
    return {"llm": "LLM", "lstar": "L*", "ttt": "TTT"}.get(strategy, strategy)


def _normalize_strategy_name(name: Any) -> Optional[str]:
    s = str(name).upper().replace(" ", "")
    if "LSTAR" in s or "L*" in s:
        return "lstar"
    if "TTT" in s:
        return "ttt"
    if "LLM" in s:
        return "llm"
    return None


def _collect_hypotheses(owner: Any) -> Dict[str, List[Dict[str, Any]]]:
    by_strategy: Dict[str, List[Dict[str, Any]]] = {"llm": [], "lstar": [], "ttt": []}

    for order, item in enumerate(getattr(owner, "eq_dfa_guesses", []) or [], start=1):
        try:
            step, candidate, witness = item
        except Exception:
            continue
        by_strategy["llm"].append({"step": int(step), "order": order, "dfa": candidate, "witness": witness})

    if not _compute_baselines_enabled():
        return by_strategy

    target = getattr(getattr(owner, "game", None), "dfa", None)
    runs = getattr(target, "strategy_results", None)
    latest = runs[-1] if isinstance(runs, list) and runs else None
    if isinstance(latest, dict):
        for name, result in latest.items():
            strategy = _normalize_strategy_name(name)
            if strategy not in {"lstar", "ttt"}:
                continue
            eq_order = 0
            for idx, hist_item in enumerate(getattr(result, "history", []) or [], start=1):
                if not isinstance(hist_item, (tuple, list)) or not hist_item:
                    continue
                if str(hist_item[0]).strip().replace("-", "_").lower() != "eq":
                    continue
                hypothesis = hist_item[4] if len(hist_item) >= 5 else None
                if hypothesis is None:
                    continue
                eq_order += 1
                by_strategy[strategy].append({"step": idx, "order": eq_order, "dfa": hypothesis, "witness": None})

    return by_strategy


def build_language_similarity_analysis(owner: Any) -> Dict[str, Any]:
    cached = getattr(owner, "_language_similarity_analysis_cache", None)
    if isinstance(cached, dict):
        return cached

    target_dfa = getattr(getattr(owner, "game", None), "dfa", None)
    by_strategy = {"llm": [], "lstar": [], "ttt": []}
    if target_dfa is None:
        return {"items_by_strategy": by_strategy, "csv_lists": _empty_csv_lists(owner), "k": _language_similarity_k()}

    hypotheses = _collect_hypotheses(owner)
    for strategy, items in hypotheses.items():
        if strategy in {"lstar", "ttt"} and not _compute_baselines_enabled():
            continue
        for item in items:
            step = int(item["step"])
            try:
                stats = compute_symdiff_similarity_stats(target_dfa, item["dfa"])
                detail_path = _write_similarity_detail_html(
                    strategy=strategy,
                    step=step,
                    target_dfa=target_dfa,
                    hypothesis_dfa=item["dfa"],
                    stats=stats,
                )
                by_strategy[strategy].append({
                    "strategy": strategy,
                    "step": step,
                    "order": item.get("order"),
                    "similarity": _format_decimal_csv(stats["similarity_decimal"]),
                    "similarity_short": _format_decimal_short(stats["similarity_decimal"]),
                    "distance": _format_decimal_csv(stats["distance_ratio_decimal"]),
                    "detail_path": detail_path,
                    "detail_url": _as_file_url(detail_path),
                    "stats": stats,
                })
            except Exception as exc:
                by_strategy[strategy].append({
                    "strategy": strategy,
                    "step": step,
                    "order": item.get("order"),
                    "similarity": "-1",
                    "similarity_short": "X",
                    "error": str(exc),
                })

    result = {
        "items_by_strategy": by_strategy,
        "csv_lists": _build_csv_lists(owner, by_strategy),
        "k": _language_similarity_k(),
    }
    owner._language_similarity_analysis_cache = result
    return result


def _max_llm_step(owner: Any) -> int:
    vals = []
    try:
        vals.append(int(getattr(owner, "_call_counter", 0) or 0))
    except Exception:
        pass
    for m in getattr(owner, "memory", []) or []:
        raw = m.get("raw") if isinstance(m, dict) else None
        outs = raw.get("tool_outputs", []) if isinstance(raw, dict) else []
        if not isinstance(outs, list):
            continue
        for out in outs:
            try:
                vals.append(int(out.get("call_count", 0) or 0))
            except Exception:
                pass
    return max(vals, default=0)


def _strategy_history_len(owner: Any, strategy: str) -> int:
    if strategy == "llm":
        return _max_llm_step(owner)
    target = getattr(getattr(owner, "game", None), "dfa", None)
    runs = getattr(target, "strategy_results", None)
    latest = runs[-1] if isinstance(runs, list) and runs else None
    if isinstance(latest, dict):
        for name, result in latest.items():
            if _normalize_strategy_name(name) == strategy:
                try:
                    return len(getattr(result, "history", []) or [])
                except Exception:
                    return 0
    return 0


def _empty_csv_lists(owner: Any) -> Dict[str, List[Any]]:
    return {s: [-1] * _strategy_history_len(owner, s) for s in ("llm", "lstar", "ttt")}


def _build_csv_lists(owner: Any, by_strategy: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Any]]:
    lists = _empty_csv_lists(owner)
    for strategy, items in by_strategy.items():
        values = lists.setdefault(strategy, [])
        for item in items:
            try:
                step = int(item.get("step") or 0)
            except Exception:
                continue
            while len(values) < step:
                values.append(-1)
            values[step - 1] = item.get("similarity", "-1")
    return lists


def language_similarity_columns_for_csv(owner: Any) -> Dict[str, str]:
    analysis = build_language_similarity_analysis(owner)
    lists = analysis.get("csv_lists", {})
    return {
        LLM_SIMILARITY_COLUMN: json.dumps(lists.get("llm", []), ensure_ascii=False),
        LSTAR_SIMILARITY_COLUMN: json.dumps(lists.get("lstar", []), ensure_ascii=False),
        TTT_SIMILARITY_COLUMN: json.dumps(lists.get("ttt", []), ensure_ascii=False),
    }




def render_language_similarity_graph_html(analysis: Optional[Dict[str, Any]], strategy: str) -> str:
    """Return a collapsed per-strategy HTML graph of similarity across game steps."""
    if not isinstance(analysis, dict):
        return ""

    items = []
    for item in analysis.get("items_by_strategy", {}).get(strategy, []) or []:
        try:
            step = int(item.get("step") or 0)
        except Exception:
            continue
        if step <= 0:
            continue
        sim_raw = item.get("similarity")
        if sim_raw in (None, "", "-1"):
            continue
        try:
            sim = float(sim_raw)
        except Exception:
            continue
        sim = max(0.0, min(1.0, sim))
        items.append((step, sim, item))

    if not items:
        return ""

    items.sort(key=lambda x: x[0])
    k = analysis.get("k", _language_similarity_k())
    width = 720
    height = 240
    left = 48
    right = 18
    top = 18
    bottom = 42
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_step = min(step for step, _sim, _item in items)
    max_step = max(step for step, _sim, _item in items)

    def x_for(step: int) -> float:
        if max_step == min_step:
            return left + plot_w / 2
        return left + ((step - min_step) / (max_step - min_step)) * plot_w

    def y_for(sim: float) -> float:
        return top + (1.0 - sim) * plot_h

    poly_points = " ".join(f"{x_for(step):.2f},{y_for(sim):.2f}" for step, sim, _item in items)

    point_nodes = []
    for step, sim, item in items:
        x = x_for(step)
        y = y_for(sim)
        label = f"step {step}: sim={_format_decimal_short(Decimal(str(sim)))}"
        detail_url = item.get("detail_url") or item.get("detail_path") or ""
        circle = (
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='4.5' class='similarity_point'>"
            f"<title>{html.escape(label)}</title></circle>"
        )
        if detail_url:
            href = html.escape(str(detail_url), quote=True)
            point_nodes.append(f"<a href='{href}' target='_blank' rel='noopener noreferrer'>{circle}</a>")
        else:
            point_nodes.append(circle)

    def tick_y(value: float) -> str:
        y = y_for(value)
        return (
            f"<line x1='{left}' y1='{y:.2f}' x2='{width-right}' y2='{y:.2f}' class='similarity_grid_line'/>"
            f"<text x='{left-8}' y='{y+4:.2f}' text-anchor='end' class='similarity_axis_label'>{value:g}</text>"
        )

    y_ticks = "".join(tick_y(v) for v in [0.0, 0.25, 0.5, 0.75, 1.0])
    x_labels = (
        f"<text x='{left}' y='{height-14}' text-anchor='middle' class='similarity_axis_label'>{min_step}</text>"
        f"<text x='{width-right}' y='{height-14}' text-anchor='middle' class='similarity_axis_label'>{max_step}</text>"
    )
    if max_step != min_step:
        mid_step = (min_step + max_step) / 2
        mid_x = (left + width - right) / 2
        x_labels += f"<text x='{mid_x:.2f}' y='{height-14}' text-anchor='middle' class='similarity_axis_label'>{mid_step:.1f}</text>"

    latest_step, latest_sim, _latest_item = items[-1]
    summary = html.escape(
        f"Similarity over game (k={k}, latest step {latest_step}: {_format_decimal_short(Decimal(str(latest_sim)))})"
    )

    return f"""
<details class="similarity_graph_box">
  <summary>{summary}</summary>
  <div class="similarity_graph_inner">
    <svg viewBox="0 0 {width} {height}" class="similarity_svg" role="img" aria-label="Similarity over game">
      <rect x="0" y="0" width="{width}" height="{height}" class="similarity_svg_bg"/>
      {y_ticks}
      <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="similarity_axis"/>
      <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="similarity_axis"/>
      {x_labels}
      <text x="{width/2:.2f}" y="{height-2}" text-anchor="middle" class="similarity_axis_label">step</text>
      <text x="14" y="{height/2:.2f}" text-anchor="middle" class="similarity_axis_label" transform="rotate(-90 14 {height/2:.2f})">similarity</text>
      <polyline points="{poly_points}" class="similarity_line"/>
      {''.join(point_nodes)}
    </svg>
  </div>
</details>
"""

def similarity_link_html(analysis: Optional[Dict[str, Any]], strategy: str, step: Any) -> str:
    if not isinstance(analysis, dict):
        return ""
    try:
        step_int = int(step)
    except Exception:
        return ""
    for item in analysis.get("items_by_strategy", {}).get(strategy, []) or []:
        if int(item.get("step") or -1) != step_int:
            continue
        if item.get("similarity") == "-1":
            return " <span class='sim_metric_link'>| sim=X</span>"
        href = html.escape(str(item.get("detail_url", "#")), quote=True)
        text = html.escape(str(item.get("similarity_short", item.get("similarity", ""))))
        return f" <a class='sim_metric_link' href='{href}' target='_blank' rel='noopener noreferrer'>| sim={text}</a>"
    return ""

# ---------------------------------------------------------------------
# Runtime/background API
# ---------------------------------------------------------------------

from concurrent.futures import ThreadPoolExecutor, Future
import threading


def _language_similarity_workers() -> int:
    try:
        from constants import LANGUAGE_SIMILARITY_BACKGROUND_WORKERS
        return max(1, int(LANGUAGE_SIMILARITY_BACKGROUND_WORKERS))
    except Exception:
        return 1


def _ensure_runtime_state(owner: Any) -> Dict[str, Any]:
    state = getattr(owner, "_language_similarity_runtime_state", None)
    if isinstance(state, dict):
        return state

    state = {
        "executor": ThreadPoolExecutor(max_workers=_language_similarity_workers()),
        "futures": [],
        "items_by_strategy": {"llm": [], "lstar": [], "ttt": []},
        "seen": set(),
        "lock": threading.Lock(),
        "order_counter": {"llm": 0, "lstar": 0, "ttt": 0},
        "closed": False,
    }
    owner._language_similarity_runtime_state = state
    return state


def initialize_runtime_language_similarity(owner: Any) -> None:
    """Prepare background language-similarity computation for a new game."""
    if hasattr(owner, "_language_similarity_analysis_cache"):
        try:
            delattr(owner, "_language_similarity_analysis_cache")
        except Exception:
            pass
    _ensure_runtime_state(owner)


def _compute_similarity_item_for_runtime(
    target_dfa: Any,
    strategy: str,
    step: int,
    order: int,
    hypothesis_dfa: Any,
) -> Dict[str, Any]:
    try:
        stats = compute_symdiff_similarity_stats(target_dfa, hypothesis_dfa)
        detail_path = _write_similarity_detail_html(
            strategy=strategy,
            step=step,
            target_dfa=target_dfa,
            hypothesis_dfa=hypothesis_dfa,
            stats=stats,
        )
        return {
            "strategy": strategy,
            "step": int(step),
            "order": order,
            "similarity": _format_decimal_csv(stats["similarity_decimal"]),
            "similarity_short": _format_decimal_short(stats["similarity_decimal"]),
            "distance": _format_decimal_csv(stats["distance_ratio_decimal"]),
            "detail_path": detail_path,
            "detail_url": _as_file_url(detail_path),
            "stats": stats,
        }
    except Exception as exc:
        return {
            "strategy": strategy,
            "step": int(step),
            "order": order,
            "similarity": "-1",
            "similarity_short": "X",
            "error": str(exc),
        }


def schedule_runtime_language_similarity(
    owner: Any,
    *,
    strategy: str,
    step: int,
    hypothesis_dfa: Any,
) -> None:
    """Queue one similarity job without blocking the game."""
    if strategy in {"lstar", "ttt"} and not _compute_baselines_enabled():
        return

    target_dfa = getattr(getattr(owner, "game", None), "dfa", None)
    if target_dfa is None or hypothesis_dfa is None:
        return

    state = _ensure_runtime_state(owner)
    key = (strategy, int(step), id(hypothesis_dfa))
    with state["lock"]:
        if state.get("closed"):
            return
        if key in state["seen"]:
            return
        state["seen"].add(key)
        state["order_counter"][strategy] = int(state["order_counter"].get(strategy, 0)) + 1
        order = state["order_counter"][strategy]
        fut = state["executor"].submit(
            _compute_similarity_item_for_runtime,
            target_dfa,
            strategy,
            int(step),
            order,
            hypothesis_dfa,
        )
        state["futures"].append(fut)




def _extract_tool_reply_eq_call(tool_reply: Dict[str, Any]) -> Optional[int]:
    outs = tool_reply.get("tool_outputs", []) if isinstance(tool_reply, dict) else []
    if not isinstance(outs, list):
        return None
    for out in outs:
        if not isinstance(out, dict):
            continue
        if str(out.get("tool_name", "")).strip() != "evaluate_dfa_candidate":
            continue
        try:
            return int(float(str(out.get("call_count", "")).strip()))
        except Exception:
            return None
    return None


def print_language_similarity_ui_snapshot(owner: Any, tool_reply: Dict[str, Any]) -> None:
    """Print one parseable launcher line for each EQ hypothesis similarity.

    This is intentionally synchronous and only runs for the current EQ tool
    call, so the launcher can show the similarity box immediately next to the
    hypothesis proposal.
    """
    call = _extract_tool_reply_eq_call(tool_reply)
    if call is None:
        return

    target_dfa = getattr(getattr(owner, "game", None), "dfa", None)
    candidate = None
    for item in getattr(owner, "eq_dfa_guesses", []) or []:
        try:
            step, dfa, _witness = item
        except Exception:
            continue
        try:
            if int(step) == int(call):
                candidate = dfa
                break
        except Exception:
            continue

    payload: Dict[str, Any]
    if target_dfa is None or candidate is None:
        payload = {"available": False, "similarity": "X", "similarity_short": "X", "error": "missing target or candidate DFA"}
    else:
        try:
            stats = compute_symdiff_similarity_stats(target_dfa, candidate)
            detail_path = _write_similarity_detail_html(
                strategy="llm",
                step=int(call),
                target_dfa=target_dfa,
                hypothesis_dfa=candidate,
                stats=stats,
            )
            payload = {
                "available": True,
                "metric": "symmetric-difference similarity",
                "k": stats.get("k"),
                "similarity": _format_decimal_csv(stats["similarity_decimal"]),
                "similarity_short": _format_decimal_short(stats["similarity_decimal"]),
                "distance": _format_decimal_csv(stats["distance_ratio_decimal"]),
                "different_words": stats.get("symmetric_difference_count"),
                "total_words": stats.get("total_words_up_to_k"),
                "detail_path": detail_path,
                "detail_url": _as_file_url(detail_path),
            }
        except Exception as exc:
            payload = {"available": False, "similarity": "X", "similarity_short": "X", "error": f"{type(exc).__name__}: {exc}"}

    print(
        "LANGUAGE_SIMILARITY_ANALYSIS::CALL="
        + str(call)
        + "::JSON="
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
        flush=True,
    )


def update_runtime_language_similarity_from_eq_guesses(owner: Any) -> None:
    """Schedule new LLM EQ hypotheses that were added during the current step."""
    for item in getattr(owner, "eq_dfa_guesses", []) or []:
        try:
            step, candidate, _witness = item
        except Exception:
            continue
        schedule_runtime_language_similarity(
            owner,
            strategy="llm",
            step=int(step),
            hypothesis_dfa=candidate,
        )


def _collect_finished_runtime_items(owner: Any, *, wait: bool) -> Dict[str, List[Dict[str, Any]]]:
    state = getattr(owner, "_language_similarity_runtime_state", None)
    items_by_strategy = {"llm": [], "lstar": [], "ttt": []}
    if not isinstance(state, dict):
        return items_by_strategy

    futures = list(state.get("futures", []) or [])
    if wait:
        for fut in futures:
            try:
                item = fut.result()
                if isinstance(item, dict):
                    items_by_strategy.setdefault(str(item.get("strategy", "llm")), []).append(item)
            except Exception as exc:
                items_by_strategy["llm"].append({
                    "strategy": "llm",
                    "step": -1,
                    "similarity": "-1",
                    "similarity_short": "X",
                    "error": str(exc),
                })
        with state.get("lock", threading.Lock()):
            if not state.get("closed"):
                state["closed"] = True
                try:
                    state["executor"].shutdown(wait=True)
                except Exception:
                    pass
    else:
        for fut in futures:
            if not fut.done():
                continue
            try:
                item = fut.result()
                if isinstance(item, dict):
                    items_by_strategy.setdefault(str(item.get("strategy", "llm")), []).append(item)
            except Exception:
                pass

    # De-duplicate by strategy+step; keep the first item/order.
    for strategy, values in list(items_by_strategy.items()):
        seen_steps = set()
        clean = []
        for item in sorted(values, key=lambda x: (int(x.get("step") or 0), int(x.get("order") or 0))):
            step = int(item.get("step") or 0)
            if step in seen_steps:
                continue
            seen_steps.add(step)
            clean.append(item)
        items_by_strategy[strategy] = clean
    return items_by_strategy


def finalize_runtime_language_similarity(owner: Any) -> Dict[str, Any]:
    """Wait for runtime jobs, add optional baselines, and cache final analysis."""
    cached = getattr(owner, "_language_similarity_analysis_cache", None)
    if isinstance(cached, dict):
        return cached

    target_dfa = getattr(getattr(owner, "game", None), "dfa", None)
    by_strategy = _collect_finished_runtime_items(owner, wait=True)
    for s in ("llm", "lstar", "ttt"):
        by_strategy.setdefault(s, [])

    # Baseline hypotheses are not on the LLM critical path. Compute them here only
    # if the fixed constant allows baseline similarity metrics.
    if target_dfa is not None and _compute_baselines_enabled():
        hypotheses = _collect_hypotheses(owner)
        for strategy in ("lstar", "ttt"):
            existing_steps = {int(x.get("step") or 0) for x in by_strategy.get(strategy, [])}
            for item in hypotheses.get(strategy, []) or []:
                step = int(item.get("step") or 0)
                if step in existing_steps:
                    continue
                by_strategy[strategy].append(
                    _compute_similarity_item_for_runtime(
                        target_dfa,
                        strategy,
                        step,
                        int(item.get("order") or 0),
                        item.get("dfa"),
                    )
                )

    result = {
        "items_by_strategy": by_strategy,
        "csv_lists": _build_csv_lists(owner, by_strategy),
        "k": _language_similarity_k(),
    }
    owner._language_similarity_analysis_cache = result
    return result

# Override the earlier post-processing implementation: from now on callers use the
# runtime/background results. If runtime was not initialized for some reason, this
# still computes the same metric safely at export time.
def build_language_similarity_analysis(owner: Any) -> Dict[str, Any]:
    state = getattr(owner, "_language_similarity_runtime_state", None)
    if isinstance(state, dict):
        return finalize_runtime_language_similarity(owner)

    # Fallback for old entry points/tests that call export without running the game loop.
    target_dfa = getattr(getattr(owner, "game", None), "dfa", None)
    by_strategy = {"llm": [], "lstar": [], "ttt": []}
    if target_dfa is None:
        return {"items_by_strategy": by_strategy, "csv_lists": _empty_csv_lists(owner), "k": _language_similarity_k()}

    hypotheses = _collect_hypotheses(owner)
    for strategy, items in hypotheses.items():
        if strategy in {"lstar", "ttt"} and not _compute_baselines_enabled():
            continue
        for item in items:
            by_strategy[strategy].append(
                _compute_similarity_item_for_runtime(
                    target_dfa,
                    strategy,
                    int(item.get("step") or 0),
                    int(item.get("order") or 0),
                    item.get("dfa"),
                )
            )
    result = {"items_by_strategy": by_strategy, "csv_lists": _build_csv_lists(owner, by_strategy), "k": _language_similarity_k()}
    owner._language_similarity_analysis_cache = result
    return result


def language_similarity_columns_for_csv(owner: Any) -> Dict[str, str]:
    analysis = build_language_similarity_analysis(owner)
    lists = analysis.get("csv_lists", {})
    return {
        LLM_SIMILARITY_COLUMN: json.dumps(lists.get("llm", []), ensure_ascii=False),
        LSTAR_SIMILARITY_COLUMN: json.dumps(lists.get("lstar", []), ensure_ascii=False),
        TTT_SIMILARITY_COLUMN: json.dumps(lists.get("ttt", []), ensure_ascii=False),
    }
