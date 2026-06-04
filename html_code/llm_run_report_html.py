"""
Responsible for:

- Exporting full interactive run histories into rich HTML visualization reports
- Rendering LLM interaction traces, strategy histories, tool outputs, and reasoning metrics
- Displaying DFA visualizations, non-informative query summaries, and runtime diagnostics
- Combining LLM performance with baseline strategy results in comparative report views
- Supporting post-run analysis through comprehensive interactive HTML dashboards
"""
from __future__ import annotations
import json
import html
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from game_types import EvaluationToolInterface
from utils import normalize_tool_name
from html_code.llm_comparison_html import read_if_path
from constants import NONINFORMATIVE_SCORE_STOP
from passive_gold_runtime import get_passive_gold_analysis, render_passive_step_html, render_passive_summary_html
from hypothesis_runtime import get_hypothesis_runtime_analysis, render_hypothesis_runtime_html
from language_similarity_runtime import build_language_similarity_analysis, similarity_link_html, render_language_similarity_graph_html


def _format_strategy_history_item(owner: Any, item: Any) -> Tuple[str, Any]:
    if isinstance(item, (tuple, list)):
        if len(item) >= 1:
            kind = str(item[0])
            rest = tuple(item[1:])
            return kind, rest

    if isinstance(item, dict):
        k = item.get("kind") or item.get("type") or "event"
        data = item.get("data", None)
        if isinstance(data, list) and isinstance(k, str):
            kk = owner.normalize_tool_name(k).lower() if hasattr(owner, "normalize_tool_name") else str(k).strip().replace("-", "_").lower()
            if kk in ("mq", "eq"):
                return k, tuple(data)
        return str(k), item

    return "event", item


def _strategy_hdr_body_from_tuple(owner: Any, kind: str, rest: Tuple[Any, ...]) -> Tuple[str, str, Optional[str]]:
    normalize_tool_name = getattr(owner, "normalize_tool_name", None)
    if callable(normalize_tool_name):
        k = normalize_tool_name(kind).lower()
    else:
        k = str(kind).strip().replace("-", "_").lower()

    def b(v: Any) -> str:
        if isinstance(v, bool):
            return "True" if v else "False"
        return str(v)

    if k == "mq":
        word = rest[0] if len(rest) > 0 else ""
        accepted = rest[1] if len(rest) > 1 else ""
        hdr_txt = f"MQ({word}) → {b(accepted)}"
        hdr = html.escape(hdr_txt)
        body_obj = {"type": "mq", "data": list(rest)}
        try:
            body = json.dumps(body_obj, ensure_ascii=False, indent=2)
        except Exception:
            body = str(body_obj)
        return hdr, body, None

    if k == "eq":
        witness = rest[0] if len(rest) > 0 else None
        ok = rest[1] if len(rest) > 1 else ""
        html_path = rest[2] if len(rest) > 2 else None

        if ok is False:
            hdr_txt = f"EQ → {b(ok)} ({witness})"
        else:
            hdr_txt = f"EQ → {b(ok)}"

        hdr = html.escape(hdr_txt)
        body_obj = {"type": "eq", "data": list(rest)}
        try:
            body = json.dumps(body_obj, ensure_ascii=False, indent=2)
        except Exception:
            body = str(body_obj)

        link = html_path if isinstance(html_path, str) and html_path.strip() else None
        return hdr, body, link

    hdr = html.escape(str(kind))
    try:
        body = json.dumps(
            {"type": kind, "data": list(rest) if isinstance(rest, tuple) else rest},
            ensure_ascii=False,
            indent=2,
        )
    except Exception:
        body = str(rest)
    return hdr, body, None


def _render_strategy_history_html(
    owner: Any,
    *,
    history: Any,
    passive_gold_analysis: Optional[Dict[str, Any]] = None,
    strategy_key: str = "",
    language_similarity_analysis: Optional[Dict[str, Any]] = None,
) -> str:
    blocks: List[str] = []
    if not isinstance(history, list):
        return ""

    idx = 1
    for item in history:
        kind, rest = _format_strategy_history_item(owner, item)

        link: Optional[str] = None
        if isinstance(rest, tuple):
            hdr, _body_txt, link = _strategy_hdr_body_from_tuple(owner, kind, rest)
        else:
            hdr = html.escape(str(kind))

        hdr_left = hdr
        if link:
            safe_href = html.escape(link, quote=True)
            hdr_left = f"<a class='tool_link' href='{safe_href}' target='_blank' rel='noopener noreferrer'>{hdr_left}</a>"
        if str(kind).strip().replace("-", "_").lower() == "eq":
            hdr_left += similarity_link_html(language_similarity_analysis, strategy_key, idx)

        blocks.append(
            f"""
            <div class="msg tool">
            <div class="hdr">
                <span class="role">{hdr_left}</span>
                <span class="idx">#{idx}</span>
            </div>
            <div class="body">
                {render_passive_step_html(passive_gold_analysis or {}, strategy_key, idx)}
            </div>
            </div>
            """
        )
        idx += 1

    return "\n".join(blocks)


def export_llm_run_html(
    owner,
    html_path: str,
    *,
    seed=None,
    conversation_link="",
) -> None:
    seed_val = seed if seed is not None else owner._last_seed
    run_minute = datetime.now().strftime("%d/%m/%Y %H:%M")
    max_tool_calls = int(getattr(owner.game, "max_tool_calls", 0) or 0)
    llm_total_queries: Any = owner._compute_llm_total_queries() if owner._reached_optimal else "X"

    dfa = getattr(owner.game, "dfa", None)
    dfa_html = "<div style='padding:16px;font-family:Arial'>DFA not available</div>"
    alphabet_size = 0
    number_of_states = 0

    if dfa is not None:
        try:
            draw_out = dfa.draw()
            dfa_html = read_if_path(draw_out)
        except Exception as e:
            dfa_html = f"<div style='padding:16px;font-family:Arial'>draw() failed: {html.escape(str(e))}</div>"

        alpha = getattr(dfa, "input_symbols", None)
        st = getattr(dfa, "states", None)
        try:
            alphabet_size = len(alpha) if alpha is not None else 0
        except Exception:
            alphabet_size = 0
        try:
            number_of_states = len(st) if st is not None else 0
        except Exception:
            number_of_states = 0

    tools = []
    for t in (getattr(owner.game, "tools", None) or []):
        tools.append(getattr(t, "tool_name", t.__class__.__name__))
    tools_str = ";".join(tools)

    hints = list(getattr(owner.game, "hints", None) or [])
    hints_str = ";".join(hints)

    llm_mq, llm_eq = owner._collect_llm_mq_eq_counts()

    title = f"LLM Strategy - {llm_total_queries}"

    try:
        passive_gold_analysis = get_passive_gold_analysis(owner, include_baselines=True)
    except Exception as exc:
        passive_gold_analysis = {"stats": {}, "steps_by_strategy": {}}
        print(f"Passive gold HTML analysis failed: {exc}")

    try:
        language_similarity_analysis = build_language_similarity_analysis(owner)
    except Exception as exc:
        language_similarity_analysis = {"items_by_strategy": {}, "csv_lists": {}}
        print(f"Language similarity HTML analysis failed: {exc}")

    try:
        hypothesis_runtime_analysis = get_hypothesis_runtime_analysis(owner)
    except Exception as exc:
        hypothesis_runtime_analysis = {}
        print(f"Hypothesis runtime HTML analysis failed: {exc}")

    total_game_time_s = getattr(owner, "_last_game_time_s", None)
    if not isinstance(total_game_time_s, (int, float)):
        total_game_time_s = ""

    game_token_tuple_obj = getattr(owner, "_last_game_token_tuple", None)
    game_token_tuple: Dict[str, Any] = game_token_tuple_obj if isinstance(game_token_tuple_obj, dict) else {}

    try:
        total_tokens_used_str = json.dumps(game_token_tuple, ensure_ascii=False, default=str)
    except Exception:
        total_tokens_used_str = str(game_token_tuple)

    cache_data_obj = getattr(owner, "_last_cache_data", None)
    cache_data: Dict[str, Any] = cache_data_obj if isinstance(cache_data_obj, dict) else {}
    cache_data_filtered = {}
    for k, v in cache_data.items():
        if str(k) == "token_counts":
            continue
        cache_data_filtered[k] = v
    try:
        cache_data_json = json.dumps(cache_data_filtered, ensure_ascii=False, default=str)
    except Exception:
        cache_data_json = str(cache_data_filtered)

    llm_model_name = str(getattr(owner, "_last_llm_model_name", "") or "")

    meta_parts = [
        f"run_minute = {run_minute}",
        f"llm_model = {llm_model_name}",
        f"total_game_time_s = {'' if total_game_time_s == '' else f'{float(total_game_time_s):.4f}'}s",
        f"tokens_used = {total_tokens_used_str}",
        f"cache_data = {cache_data_json}",
        f"max_tool_calls = {max_tool_calls}",
        f"alphabet_size = {alphabet_size}",
        f"number_of_states = {number_of_states}",
        f"seed = {'' if seed_val is None else seed_val}",
        f"tools = {tools_str}",
        f"llm_hypothesis_monotonicity_broken = {int(hypothesis_runtime_analysis.get('llm_hypothesis_monotonicity_broken', 0) or 0)}",
        f"llm_eq_count_gt_target_states = {int(hypothesis_runtime_analysis.get('llm_eq_count_gt_target_states', 0) or 0)}",
        f"mq_total = {llm_mq}",
        f"eq_total = {llm_eq}",

    ]
    meta_html = "<br/>".join(html.escape(p) for p in meta_parts)

    step_metrics_obj = getattr(owner, "step_tokens", None)
    step_metrics: Dict[str, Any] = step_metrics_obj if isinstance(step_metrics_obj, dict) else {}

    step_thoughts_obj = getattr(owner, "_last_step_thought_texts", None)
    step_thoughts: Dict[str, Any] = step_thoughts_obj if isinstance(step_thoughts_obj, dict) else {}

    issue_map = owner._build_noninformative_step_reason_map()
    issue_steps = set(issue_map.keys())
    summary_dict = owner.build_run_summary_dict()

    def _safe_list_len(v: Any) -> int:
        return len(v) if isinstance(v, list) else 0

    chart_fields = [
        ("MQ DUPLICATE STEPS", "count of MQ calls that repeated a previous MQ word"),
        ("MQ HITS PREVIOUS EQ WITNESS", "count of MQ calls that queried a previous EQ witness word"),
        ("EQ DUPLICATE STEPS", "count of EQ calls whose candidate DFA was equivalent to a previous candidate"),
        ("EQ CONTRADICTS PREVIOUS MQ", "count of EQ calls that contradicted earlier MQ labels (deduped by EQ step)"),
        ("EQ CONTRADICTS PREVIOUS EQ WITNESS", "count of EQ calls that contradicted earlier EQ witness labels (deduped by EQ step)"),
    ]

    chart_data = []
    max_count = 1
    for key, desc, in chart_fields:
        c = _safe_list_len(summary_dict.get(key))
        if c > max_count:
            max_count = c
        chart_data.append((key, desc, c))

    chart_items_html = []
    for key, desc, c in chart_data:
        pct = int(round((c / max_count) * 100)) if max_count > 0 else 0
        chart_items_html.append(
            f"""
            <div class="bar_item">
            <div class="bar_label">{html.escape(key)}</div>
            <div class="bar_track">
                <div class="bar_fill" style="height:{pct}%"></div>
            </div>
            <div class="bar_value">{html.escape(str(c))}</div>
            </div>
            """
        )

    noninf_chart_html = f"""
    <div class="noninf_chart">
    <div class="chart_title">noninformative queries summary</div>
    <div class="chart_grid">
        {''.join(chart_items_html)}
    </div>
    </div>
    """

    def _step_metrics_offset() -> int:
        keys: List[int] = []
        for k in step_metrics.keys():
            try:
                keys.append(int(k))
            except Exception:
                continue
        if not keys:
            return 1
        return 0 if min(keys) == 0 else 1

    _offset = _step_metrics_offset()

    def _metrics_for_call_count(call_count: Any) -> Tuple[str, str]:
        try:
            cc = int(call_count)
        except Exception:
            return "", ""
        step_idx = (cc - 1) if _offset == 0 else cc
        m = step_metrics.get(str(step_idx))
        if not isinstance(m, dict):
            m = step_metrics.get(step_idx) if step_idx in step_metrics else None
        if not isinstance(m, dict):
            return "", ""
        t = m.get("step_time_s", "")
        th = m.get("thoughts_tokens", "")
        try:
            time_int = str(int(round(float(t))))
        except Exception:
            time_int = ""
        try:
            thoughts_int = str(int(th))
        except Exception:
            try:
                thoughts_int = str(int(float(th)))
            except Exception:
                thoughts_int = ""
        return time_int, thoughts_int

    def _thought_text_for_call_count(call_count: Any) -> str:
        try:
            cc = int(call_count)
        except Exception:
            return ""

        step_idx = (cc - 1) if _offset == 0 else cc

        raw = step_thoughts.get(str(step_idx))
        if raw is None:
            raw = step_thoughts.get(step_idx)

        if raw is None:
            return ""

        if isinstance(raw, str):
            return raw

        try:
            return json.dumps(raw, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(raw)

    def _render_llm_memory_html_with_metrics(*, initial_prompt: str) -> str:
        eval_tool_names = set()
        for t in (getattr(owner.game, "tools", None) or []):
            name = getattr(t, "tool_name", t.__class__.__name__)
            try:
                if isinstance(t, EvaluationToolInterface):
                    eval_tool_names.add(normalize_tool_name(name))
            except Exception:
                pass

        blocks: List[str] = []
        blocks.append(
            f"""
            <div class="msg prompt">
            <div class="hdr">
                <span class="role">Opening prompt in English</span>
                <span class="idx">#0</span>
            </div>
            <div class="body">
                <details class="payload">
                <summary>open</summary>
                <pre class="content">{html.escape(initial_prompt)}</pre>
                </details>
                {render_passive_summary_html(passive_gold_analysis, "llm")}
                {render_hypothesis_runtime_html(owner)}
                {render_language_similarity_graph_html(language_similarity_analysis, "llm")}
            </div>
            </div>
            """
        )

        def tool_header_html(out: Dict[str, Any]) -> Tuple[str, str, str, str]:

            tool_name = normalize_tool_name(out.get("tool_name"))
            call_count = out.get("call_count", "")
            err = out.get("error")

            payload = out.get("output") or {}
            if not isinstance(payload, dict):
                payload = {}

            time_int, thoughts_int = _metrics_for_call_count(call_count)
            suffix = ""
            if time_int or thoughts_int:
                suffix = (
                    f" <span class='tool_meta'>| time={html.escape(time_int)}s | reasoning tokens={html.escape(thoughts_int)}</span>"
                )

            subtitle = ""
            try:
                cc_int = int(call_count)
                if cc_int in issue_map:
                    subtitle = f"<div class='subhdr'>{html.escape(issue_map[cc_int])}</div>"
            except Exception:
                subtitle = ""

            if tool_name == "is_word_in_language":
                w = payload.get("word", "")
                acc = payload.get("accepted", None)
                if err:
                    return f"MQ({html.escape(str(w))}) → ERROR{suffix}{subtitle}", f"#{call_count}", call_count, tool_name
                return f"MQ({html.escape(str(w))}) → {html.escape(str(acc))}{suffix}{subtitle}", f"#{call_count}", call_count, tool_name

            if tool_name in eval_tool_names:
                optimal = payload.get("optimal", None)
                witness = payload.get("witness_word", "")
                html_path = payload.get("html", "")

                if err:
                    txt = "EQ → ERROR"
                else:
                    status = bool(optimal) if isinstance(optimal, bool) else optimal
                    if status is False and witness:
                        txt = f"EQ → {status} ({witness})"
                    else:
                        txt = f"EQ → {status}"

                sim_html = similarity_link_html(language_similarity_analysis, "llm", call_count)

                if isinstance(html_path, str) and html_path.strip():
                    safe_href = html.escape(html_path, quote=True)
                    return (
                        f"<a class='tool_link' href='{safe_href}' target='_blank' rel='noopener noreferrer'>{html.escape(txt)}{suffix}</a>{sim_html}{subtitle}",
                        f"#{call_count}",
                        call_count,
                        tool_name,
                    )

                return f"{html.escape(txt)}{suffix}{sim_html}{subtitle}", f"#{call_count}", call_count, tool_name

            base = html.escape(str(tool_name))
            if err:
                base = base + " → ERROR"
            return f"{base}{suffix}{subtitle}", f"#{call_count}", call_count, tool_name

        for m in owner.memory:
            if m.get("role") != "tool":
                continue

            raw = m.get("raw", None)
            tool_outputs = []
            if isinstance(raw, dict):
                tool_outputs = raw.get("tool_outputs", []) or []
            if not isinstance(tool_outputs, list) or not tool_outputs:
                try:
                    raw_txt = json.dumps(raw, ensure_ascii=False, indent=2)
                except Exception:
                    raw_txt = str(raw)

                blocks.append(
                    f"""
                    <div class="msg tool">
                    <div class="hdr">
                        <span class="role">tool</span>
                        <span class="idx"></span>
                    </div>
                    <div class="body">
                        <details class="payload">
                        <summary>open</summary>
                        <pre>{html.escape(raw_txt)}</pre>
                        </details>
                    </div>
                    </div>
                    """
                )
                continue

            for out in tool_outputs:
                if not isinstance(out, dict):
                    continue

                hdr_left, hdr_right, call_count, _tn = tool_header_html(out)
                hdr_style = ""
                try:
                    out_txt = json.dumps(out, ensure_ascii=False, indent=2)
                except Exception:
                    out_txt = str(out)

                eq_counts_txt = out.get("equivalence_class_counts_text", "")
                eq_counts_html = ""
                if isinstance(eq_counts_txt, str) and eq_counts_txt.strip():
                    eq_counts_html = f"""
                        <details class="payload">
                        <summary>equivalence class counts</summary>
                        <pre>{html.escape(eq_counts_txt)}</pre>
                        </details>
                    """

                thought_txt = _thought_text_for_call_count(call_count)

                thoughts_html = ""
                if thought_txt.strip():
                    thoughts_html = f"""
                        <details class="payload thought_payload">
                        <summary>model reasoning</summary>
                        <pre class="thought_text">{html.escape(thought_txt)}</pre>
                        </details>
                    """

                blocks.append(
                    f"""
                    <details class="msg tool">
                    <summary class="hdr"{hdr_style}>
                        <span class="role">{hdr_left}</span>
                        <span class="idx">{html.escape(str(hdr_right))}</span>
                    </summary>
                    <div class="body">
                        <details class="payload">
                        <summary>tool output</summary>
                        <pre>{html.escape(out_txt)}</pre>
                        </details>
                        {eq_counts_html}
                        {render_passive_step_html(passive_gold_analysis, "llm", call_count)}
                        {thoughts_html}
                    </div>
                    </details>
                    """
                )

        return "\n".join(blocks)

    initial_prompt = owner._build_initial_text()
    llm_left = _render_llm_memory_html_with_metrics(initial_prompt=initial_prompt)

    def _normalize_strategy_name_for_html(name: Any) -> str:
        upper = str(name).upper().replace(" ", "")
        if "LSTAR" in upper or "L*" in upper:
            return "lstar"
        if "TTT" in upper:
            return "ttt"
        return "llm"

    strategy_panels_html: List[str] = []
    runs = getattr(dfa, "strategy_results", None) if dfa is not None else None
    latest: Any = None
    if isinstance(runs, list) and runs:
        latest = runs[-1]

    if isinstance(latest, dict):
        for strat_name, res in latest.items():
            total_q = getattr(res, "total_queries", 0)
            mq_q = getattr(res, "mq_queries", 0)
            eq_q = getattr(res, "eq_queries", 0)
            history = getattr(res, "history", [])

            strat_title = f"{strat_name} - {total_q}"
            strat_meta = f"mq_total = {mq_q} | eq_total = {eq_q}"
            strat_key = _normalize_strategy_name_for_html(strat_name)
            strat_body = _render_strategy_history_html(
                owner,
                history=history,
                passive_gold_analysis=passive_gold_analysis,
                strategy_key=strat_key,
                language_similarity_analysis=language_similarity_analysis,
            )

            strategy_panels_html.append(
                f"""
                <details class="panel" open>
                <summary class="panel_summary">
                    <span class="panel_title">{html.escape(str(strat_title))}</span>
                </summary>
                <div class="panel_body">
                    <div class="panel_meta_open">{html.escape(str(strat_meta))}</div>
                    <div class="split_scroll">
                    {render_passive_summary_html(passive_gold_analysis, _normalize_strategy_name_for_html(strat_name))}
                    {render_language_similarity_graph_html(language_similarity_analysis, _normalize_strategy_name_for_html(strat_name))}
                    {strat_body}
                    </div>
                </div>
                </details>
                """
            )

    llm_panel_html = f"""
    <details class="panel" open>
    <summary class="panel_summary">
        <span class="panel_title">{html.escape(title)}</span>
    </summary>
    <div class="panel_body">
        <div class="panel_meta_open">{meta_html}</div>
        <div class="split_scroll">
        {llm_left}
        </div>
    </div>
    </details>
    """

    page_html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)}</title>
<style>
html, body {{ height: 100%; margin: 0; font-family: Arial, sans-serif; background: #fff; overflow: hidden; }}

.layout {{
height: 100vh;
display: grid;
grid-template-columns: 1fr 2fr;
gap: 12px;
padding: 12px;
box-sizing: border-box;
min-height: 0;
}}

.llm_col {{
min-height: 0;
height: 100%;
overflow: auto;
}}

.right_col {{ min-height: 0; display: flex; flex-direction: column; }}

.panel {{
border: 1px solid #ddd;
border-radius: 12px;
overflow: hidden;
background: #fff;
min-height: 0;
display: flex;
flex-direction: column;
}}

.panel_summary {{
cursor: pointer;
padding: 10px 12px;
background: #f7f7f7;
display: flex;
justify-content: space-between;
gap: 12px;
align-items: baseline;
user-select: none;
flex: 0 0 auto;
}}

.panel_title {{ font-weight: 700; }}

.panel_body {{
padding: 0;
display: flex;
flex-direction: column;
flex: 1 1 auto;
min-height: 0;
}}

.panel_meta_open {{
display: none;
padding: 8px 12px;
font-size: 12px;
color: #555;
border-top: 1px solid #eee;
border-bottom: 1px solid #eee;
background: #fff;
flex: 0 0 auto;
line-height: 1.5;
}}

details[open] > .panel_body > .panel_meta_open {{ display: block; }}

.split_scroll {{
flex: 1 1 auto;
min-height: 0;
overflow-y: auto;
overflow-x: hidden;
padding: 12px;
box-sizing: border-box;
-webkit-overflow-scrolling: touch;
}}

.noninf_chart {{
margin-top: 14px;
border: 1px solid #ddd;
border-radius: 12px;
padding: 10px;
}}

.chart_title {{
font-weight: 700;
margin-bottom: 10px;
}}

.chart_grid {{
display: grid;
grid-template-columns: repeat(5, 1fr);
gap: 10px;
align-items: end;
}}

.bar_item {{
border: 1px solid #eee;
border-radius: 10px;
padding: 8px;
}}

.bar_label {{
font-size: 7px;
font-weight: 700;
margin-bottom: 6px;
}}

.bar_track {{
height: 140px;
border: 1px solid #eee;
border-radius: 10px;
display: flex;
align-items: flex-end;
overflow: hidden;
}}

.bar_fill {{
width: 100%;
background: #888;
}}

.bar_value {{
margin-top: 6px;
font-size: 12px;
font-weight: 700;
}}

.bar_desc {{
margin-top: 4px;
font-size: 11px;
color: #666;
line-height: 1.3;
}}

@media (max-width: 1200px) {{
.chart_grid {{ grid-template-columns: 1fr; }}
.bar_track {{ height: 110px; }}
}}

.msg {{ border: 1px solid #eee; border-radius: 10px; margin: 10px 0; overflow: visible; }}
.hdr {{ display: flex; justify-content: space-between; padding: 8px 10px; border-bottom: 1px solid #f1f1f1; background: #fafafa; }}
.role {{ font-weight: 700; }}
.idx {{ color: #777; font-size: 12px; }}
.body {{ padding: 10px; }}
pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; overflow: visible; }}
.content {{ font-size: 13px; }}
details.payload {{ margin: 0; }}
details.payload summary {{ cursor: pointer; font-size: 12px; color: #444; user-select: none; }}
.passive_gold_summary {{ margin-top: 8px; }}
.passive_gold_table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 6px; }}
.passive_gold_table td {{ border: 1px solid #eee; padding: 4px 6px; }}
.passive_box {{ margin-top: 6px; border: 1px solid #ddd; border-radius: 8px; padding: 8px; font-size: 12px; }}
.passive_observations {{ margin-top: 8px; }}
.passive_obs_group {{ margin: 6px 0; }}
.passive_obs_title {{ font-weight: 700; margin-bottom: 4px; }}
.passive_obs_words {{ display: flex; flex-wrap: wrap; gap: 4px; }}
.passive_obs_words code {{ border: 1px solid #e5e7eb; border-radius: 6px; padding: 1px 5px; background: #fff; }}
.passive_obs_empty {{ color: #777; font-style: italic; }}
.passive_ok {{ background: #f0fdf4; }}
.passive_bad {{ background: #fff7ed; }}
.passive_unknown {{ background: #f8fafc; }}
.msg.tool .hdr {{ background: #fff8f2; }}
.msg.prompt .hdr {{ background: #f6fbff; }}
a.tool_link {{ color: inherit; text-decoration: none; }}
a.tool_link:hover {{ text-decoration: underline; }}

.tool_meta {{ color: #888; font-size: 12px; font-weight: 400; }}
.subhdr {{ font-size: 6px; font-weight: 400; margin-top: 2px; }}

.llm_col > details.panel {{ min-height: 0; }}
.llm_col > details.panel > .panel_body {{ min-height: 0; }}

.strategy_scroll {{
flex: 0 0 50vh;
min-height: 0;
overflow: auto;
}}

.strategy_grid {{
min-height: 0;
display: grid;
grid-template-columns: repeat(2, 1fr);
grid-auto-rows: max-content;
gap: 12px;
align-content: start;
padding: 0;
box-sizing: border-box;
overflow: visible;
}}

.dfa_strip {{
flex: 1 1 auto;
min-height: 0;
border-top: 1px solid #ddd;
box-sizing: border-box;
overflow: auto;
padding: 10px;
}}

.dfa_strip iframe {{ width: 100%; height: 70vh; border: 0; }}

.msg.tool summary.hdr {{
list-style: none;
cursor: pointer;
}}

.msg.tool summary.hdr::-webkit-details-marker {{
display: none;
}}

.thought_payload {{
margin-top: 8px;
}}

.thought_text {{
background: #f9f9ff;
border: 1px solid #e6e6f5;
border-radius: 8px;
padding: 10px;
font-size: 12px;
line-height: 1.45;
}}

@media (max-width: 1200px) {{
.layout {{ grid-template-columns: 1fr; }}
.strategy_grid {{ grid-template-columns: 1fr; }}
.strategy_scroll {{ flex: 0 0 45vh; }}
}}

.similarity_graph_box {{
margin-top: 10px;
border: 1px solid #ddd;
border-radius: 10px;
background: #fff;
font-size: 12px;
}}
.similarity_graph_box > summary {{
cursor: pointer;
padding: 8px 10px;
font-weight: 700;
user-select: none;
}}
.similarity_graph_inner {{
padding: 8px 10px 10px 10px;
border-top: 1px solid #eee;
}}
.similarity_svg {{
width: 100%;
height: auto;
display: block;
}}
.similarity_svg_bg {{
fill: #fff;
}}
.similarity_grid_line {{
stroke: #eeeeee;
stroke-width: 1;
}}
.similarity_axis {{
stroke: #555;
stroke-width: 1.2;
}}
.similarity_axis_label {{
fill: #666;
font-size: 11px;
font-family: Arial, sans-serif;
}}
.similarity_line {{
fill: none;
stroke: #444;
stroke-width: 2.2;
stroke-linejoin: round;
stroke-linecap: round;
}}
.similarity_point {{
fill: #fff;
stroke: #444;
stroke-width: 2;
}}
.similarity_point:hover {{
fill: #f3f4f6;
}}

.sim_metric_link {{
color: #888 !important;
font-size: 12px !important;
font-weight: 400 !important;
margin-left: 4px;
text-decoration: none !important;
}}
.sim_metric_link:hover {{ text-decoration: underline !important; }}
</style>
</head>
<body>
<div class="layout">
<div class="llm_col">
    {llm_panel_html}
</div>

<div class="right_col">
    <div class="strategy_scroll">
        <div class="strategy_grid">
        {''.join(strategy_panels_html)}
        </div>
    </div>

    <div class="dfa_strip">
    <iframe id="dfaFrame" srcdoc="{html.escape(dfa_html, quote=True)}"></iframe>
    {(noninf_chart_html)}
    </div>
</div>
</div>

<script>
(function() {{
const f = document.getElementById("dfaFrame");
if (!f) return;
f.addEventListener("load", () => {{
    try {{
    const doc = f.contentDocument || (f.contentWindow && f.contentWindow.document);
    if (!doc) return;
    const root = doc.scrollingElement || doc.documentElement || doc.body;
    const target = Math.max(0, Math.floor((root.scrollHeight - f.clientHeight) / 2));
    f.contentWindow.scrollTo(0, target);
    }} catch (e) {{}}
}});
}})();
</script>

</body>
</html>
"""

    out_path = Path(html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page_html, encoding="utf-8")