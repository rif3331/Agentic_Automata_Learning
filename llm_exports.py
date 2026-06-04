from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from html_code.llm_run_report_html import export_llm_run_html
from passive_gold_runtime import PASSIVE_CSV_COLUMNS, passive_columns_for_csv
from language_similarity_runtime import (
    LANGUAGE_SIMILARITY_CSV_COLUMNS,
    language_similarity_columns_for_csv,
)
from hypothesis_runtime import HYPOTHESIS_CSV_COLUMNS, hypothesis_columns_for_csv


CSV_DIALECT = {
    "quoting": csv.QUOTE_ALL,
    "lineterminator": "\n",
}

# Keep every CSV cell as one physical line. Some editors/viewers split rows
# when a quoted cell contains real newline characters, so we never write real
# newlines inside a field.
_REAL_LINE_SEPARATORS = re.compile(
    r"\r\n|\r|\n|\x0b|\x0c|\x1c|\x1d|\x1e|\u0085|\u2028|\u2029"
)


def _single_line(value: Any) -> str:
    if value is None:
        return ""
    return _REAL_LINE_SEPARATORS.sub(r"\\n", str(value))


def _json_safe_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe_obj(v) for v in obj]
    if isinstance(obj, str):
        return _single_line(obj)
    return obj


def make_csv_safe_json(obj: Any, *, default: Any = str) -> str:
    try:
        return _single_line(
            json.dumps(
                _json_safe_obj(obj),
                ensure_ascii=False,
                default=default,
                separators=(",", ":"),
            )
        )
    except Exception:
        return _single_line(obj)


def make_csv_safe_text(value: Any) -> str:
    return _single_line(value)


def make_csv_safe_cell(value: Any) -> str:
    """Return a value guaranteed to occupy exactly one CSV cell and one physical line."""
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return make_csv_safe_json(value, default=str)
    return _single_line(value)


def dfa_to_csv_text(dfa: Any) -> str:
    """Serialize a DFA/automaton exactly as text for one CSV cell."""
    if dfa is None:
        return ""
    return make_csv_safe_text(str(dfa))


def hypothesis_automata_dict_from_llm(owner: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    guesses = getattr(owner, "eq_dfa_guesses", None)
    if not isinstance(guesses, list):
        return out
    for i, item in enumerate(guesses, start=1):
        try:
            cand = item[1]
        except Exception:
            continue
        out[str(i)] = dfa_to_csv_text(cand)
    return out


def hypothesis_automata_dict_from_strategy_result(result: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    history = getattr(result, "history", None)
    if not isinstance(history, list):
        return out
    idx = 0
    for item in history:
        try:
            kind = item[0]
        except Exception:
            continue
        if kind != "EQ":
            continue
        automaton = None
        if isinstance(item, (tuple, list)) and len(item) >= 5:
            automaton = item[4]
        if automaton is None:
            continue
        idx += 1
        out[str(idx)] = dfa_to_csv_text(automaton)
    return out


def get_strategy_hypothesis_automata(owner: Any, strategy_name_contains: str) -> Dict[str, str]:
    dfa = getattr(getattr(owner, "game", None), "dfa", None)
    runs = getattr(dfa, "strategy_results", None)
    if not isinstance(runs, list) or not runs:
        return {}
    last = runs[-1]
    if not isinstance(last, dict):
        return {}
    wanted = strategy_name_contains.lower()
    for name, result in last.items():
        if wanted in str(name).lower():
            return hypothesis_automata_dict_from_strategy_result(result)
    return {}


def make_file_link(path: str) -> str:
    try:
        return Path(os.path.abspath(path)).as_uri()
    except Exception:
        return os.path.abspath(path)


def count_csv_rows(csv_path: str) -> int:
    try:
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            return sum(1 for _ in csv.reader(f))
    except Exception:
        return 0


def read_existing_csv_header(csv_path: str) -> Optional[List[str]]:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None
    try:
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            r = csv.reader(f)
            hdr = next(r, None)
            if not hdr:
                return None
            return [h.strip() for h in hdr]
    except Exception:
        return None


def _read_existing_rows_as_dicts(csv_path: str, header: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return rows
    try:
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            dr = csv.DictReader(f, restkey="__extra__", restval="")
            for raw_row in dr:
                row: Dict[str, Any] = {}
                for col in header:
                    row[col] = raw_row.get(col, "")
                extras = raw_row.get("__extra__")
                if extras:
                    # Preserve malformed extra cells rather than dropping them silently.
                    last_col = header[-1]
                    row[last_col] = str(row.get(last_col, "")) + " " + " ".join(map(str, extras))
                rows.append(row)
    except Exception:
        return []
    return rows


def _rewrite_csv_single_line(csv_path: str, header: List[str]) -> None:
    """Normalize the existing CSV so old multiline cells are rewritten as one-line cells too."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return

    rows = _read_existing_rows_as_dicts(csv_path, header)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=header,
            extrasaction="ignore",
            **CSV_DIALECT,
        )
        w.writeheader()
        for row in rows:
            safe_row = {col: make_csv_safe_cell(row.get(col, "")) for col in header}
            w.writerow(safe_row)


def upgrade_csv_if_needed(csv_path: str, new_columns: List[str]) -> List[str]:
    old_header = read_existing_csv_header(csv_path)

    if old_header is None:
        return list(new_columns)

    final_header = list(old_header)
    for col in new_columns:
        if col not in final_header:
            final_header.append(col)

    # Do not rewrite or normalize existing rows.
    # Only the new row written by export_run_to_csv is sanitized.
    return final_header


def export_run_to_csv(
    owner: Any,
    csv_path: str,
    *,
    seed: Optional[int] = None,
    conversation_link: str = "",
) -> None:
    run_minute = datetime.now().strftime("%d/%m/%Y %H:%M")
    max_tool_calls = int(getattr(owner.game, "max_tool_calls", 0) or 0)

    llm_total_queries: Any = (
        owner._compute_llm_total_queries()
        if getattr(owner, "_reached_optimal", False)
        else "X"
    )

    dfa = getattr(owner.game, "dfa", None)
    counterexample_mode = ""
    alphabet_size = 0
    number_of_states = 0

    if dfa is not None:
        counterexample_mode_raw = getattr(dfa, "counterexample_mode", "")
        counterexample_mode_map = {
            0: "deterministic short counterexample",
            1: "minimal counterexample",
            2: "counts based counterexample",
        }
        counterexample_mode = counterexample_mode_map.get(
            counterexample_mode_raw,
            str(counterexample_mode_raw),
        )

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
    hints_str = ";".join(map(str, hints))

    strategies = owner._get_strategies_summary()
    game_mode = str(getattr(owner, "_last_game_mode", "") or "")

    total_game_time_s = getattr(owner, "_last_game_time_s", None)
    if not isinstance(total_game_time_s, (int, float)):
        total_game_time_s = ""

    game_token_tuple_obj = getattr(owner, "_last_game_token_tuple", None)
    game_token_tuple_json = (
        make_csv_safe_json(
            game_token_tuple_obj,
            default=getattr(owner, "_json_default", str),
        )
        if isinstance(game_token_tuple_obj, dict)
        else ""
    )

    step_metrics_obj = getattr(owner, "step_tokens", None)
    step_metrics_json = (
        make_csv_safe_json(step_metrics_obj, default=str)
        if isinstance(step_metrics_obj, dict)
        else ""
    )

    cache_data_obj = getattr(owner, "_last_cache_data", None)
    cache_data_json = (
        make_csv_safe_json(cache_data_obj, default=str)
        if isinstance(cache_data_obj, dict)
        else ""
    )

    try:
        noninf_info_obj = owner.build_run_summary_dict()
    except Exception:
        noninf_info_obj = {}

    noninf_info_json = make_csv_safe_json(
        noninf_info_obj,
        default=getattr(owner, "_json_default", str),
    )

    full_last_model_context = make_csv_safe_text(
        getattr(owner, "_last_full_model_context_text", "")
    )

    target_automaton = dfa_to_csv_text(dfa)
    llm_hypothesis_automata = make_csv_safe_json(
        hypothesis_automata_dict_from_llm(owner),
        default=str,
    )
    lstar_hypothesis_automata = make_csv_safe_json(
        get_strategy_hypothesis_automata(owner, "lstar"),
        default=str,
    )
    ttt_hypothesis_automata = make_csv_safe_json(
        get_strategy_hypothesis_automata(owner, "ttt"),
        default=str,
    )

    header = [
        "run_minute",
        "llm_model",
        "game_mode",
        "max_tool_calls",
        "llm_total_queries",
        "alphabet_size",
        "number_of_states",
        "seed",
        "counterexample_mode",
        "conversation_link",
        "tools",
        "hints",
        "strategies",
        "total_game_time_s",
        "game_token_tuple",
        "step_metrics",
        "cache_data",
        "noninformative_queries_info",
        "target_automaton",
        "llm_hypothesis_automata",
        "lstar_hypothesis_automata",
        "ttt_hypothesis_automata",
        "full_last_model_context",
        *PASSIVE_CSV_COLUMNS,
        *LANGUAGE_SIMILARITY_CSV_COLUMNS,
        *HYPOTHESIS_CSV_COLUMNS,
    ]

    header = upgrade_csv_if_needed(csv_path, header)

    row: Dict[str, Any] = {
        "run_minute": run_minute,
        "llm_model": getattr(owner, "_last_llm_model_name", ""),
        "game_mode": game_mode,
        "max_tool_calls": max_tool_calls,
        "llm_total_queries": llm_total_queries,
        "alphabet_size": alphabet_size,
        "number_of_states": number_of_states,
        "seed": "" if seed is None else seed,
        "counterexample_mode": counterexample_mode,
        "conversation_link": conversation_link,
        "tools": tools_str,
        "hints": hints_str,
        "strategies": strategies,
        "total_game_time_s": total_game_time_s,
        "game_token_tuple": game_token_tuple_json,
        "step_metrics": step_metrics_json,
        "cache_data": cache_data_json,
        "noninformative_queries_info": noninf_info_json,
        "target_automaton": target_automaton,
        "llm_hypothesis_automata": llm_hypothesis_automata,
        "lstar_hypothesis_automata": lstar_hypothesis_automata,
        "ttt_hypothesis_automata": ttt_hypothesis_automata,
        "full_last_model_context": full_last_model_context,
    }

    try:
        row.update(passive_columns_for_csv(owner))
    except Exception as exc:
        for col in PASSIVE_CSV_COLUMNS:
            row[col] = "FALSE" if col.endswith("reached_gold_triangle") else "-1"
        print(f"Passive gold runtime analysis failed: {exc}")

    try:
        row.update(language_similarity_columns_for_csv(owner))
    except Exception as exc:
        for col in LANGUAGE_SIMILARITY_CSV_COLUMNS:
            row[col] = "[]"
        print(f"Language similarity analysis failed: {exc}")

    try:
        row.update(hypothesis_columns_for_csv(owner))
    except Exception as exc:
        for col in HYPOTHESIS_CSV_COLUMNS:
            row[col] = "0"
        print(f"Hypothesis runtime analysis failed: {exc}")

    if bool(getattr(owner, "_run_crashed_mid_game", False)):
        row = {col: "None" for col in header}
        row["total_game_time_s"] = total_game_time_s
        row["full_last_model_context"] = full_last_model_context

    # Absolute final safety pass over every column.
    # This is what guarantees one run = exactly one physical CSV line.
    row = {col: make_csv_safe_cell(row.get(col, "")) for col in header}

    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    def _write(path: str) -> None:
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(
                f,
                fieldnames=header,
                extrasaction="ignore",
                **CSV_DIALECT,
            )
            if need_header:
                w.writeheader()
            w.writerow(row)

    try:
        _write(csv_path)
        saved_path = os.path.abspath(csv_path)
    except Exception:
        fallback_csv = csv_path.replace(".csv", "_fallback.csv")
        # If writing to fallback, make sure header is written there if needed.
        fallback_need_header = not os.path.exists(fallback_csv) or os.path.getsize(fallback_csv) == 0
        with open(fallback_csv, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(
                f,
                fieldnames=header,
                extrasaction="ignore",
                **CSV_DIALECT,
            )
            if fallback_need_header:
                w.writeheader()
            w.writerow(row)
        saved_path = os.path.abspath(fallback_csv)

    row_number = count_csv_rows(saved_path)
    print(f"Game results were saved in row {row_number}: {make_file_link(saved_path)}")


def export_run_to_html(
    owner: Any,
    html_path: str,
    *,
    seed: Optional[int] = None,
    conversation_link: str = "",
) -> None:
    export_llm_run_html(
        owner=owner,
        html_path=html_path,
        seed=seed,
        conversation_link=conversation_link,
    )
