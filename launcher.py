from __future__ import annotations

import csv
import html as html_lib
import json
import os
import re
import shlex
import subprocess
import signal
import sys
import threading
import uuid
import contextvars
import zipfile
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote
from typing import Any, Iterable, Iterator

from flask import Flask, request, render_template_string, Response, jsonify, redirect, url_for, send_from_directory, session, send_file

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret-key-for-production")

ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_CSV = ROOT / "runs" / "results.csv"
FALLBACK_RESULTS_CSV = ROOT / "results.csv"
GLOBAL_RESULTS_CSV = ROOT / "runs" / "all_users_results.csv"
AUTO_KEY_DAILY_COSTS_CSV = ROOT / "runs" / "auto_key_daily_costs.csv"
FLASH_LITE_MODEL = "gemini-3.1-flash-lite-preview"


DEFAULT_LAST_FORM: dict[str, str] = {
    "api_provider": "google",
    "model_name": "gemini-3.1-flash-lite-preview",
    "api_key": "",
    "n_states": "2",
    "seed": "1",
    "alphabet_size": "2",
    "target_source": "regex",
    "regex": "b*a*",
    "counterexample_mode": "deterministic short counterexample",
    "algorithm_approximation_ratio": "2",
    "output_dir": "runs",
    "experiment_csv": "results.csv",
}

_current_sid: contextvars.ContextVar[str | None] = contextvars.ContextVar("launcher_current_sid", default=None)
sessions_lock = threading.RLock()
sessions: dict[str, dict[str, Any]] = {}


def _get_sid() -> str:
    sid = _current_sid.get()
    if sid:
        return sid
    sid = session.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        session["sid"] = sid
    return str(sid)


def _session_output_dir(sid: str) -> str:
    return str(Path("runs") / "sessions" / sid)


def _new_session_state(sid: str) -> dict[str, Any]:
    form = dict(DEFAULT_LAST_FORM)
    form["output_dir"] = _session_output_dir(sid)
    return {
        "logs": [],
        "running": False,
        "process": None,
        "current_target_path": "",
        "current_full_report_path": "",
        "cached_full_report_path": "",
        "last_form": form,
        "stop_flag_path": ROOT / "runs" / "sessions" / sid / "STOP_REQUESTED.flag",
        "auto_key_used": False,
        "finalized_once": False,
    }


def _state(sid: str | None = None) -> dict[str, Any]:
    sid = sid or _get_sid()
    with sessions_lock:
        if sid not in sessions:
            sessions[sid] = _new_session_state(sid)
        return sessions[sid]


class _SessionLogsProxy:
    def _logs(self) -> list[str]:
        return _state()["logs"]
    def append(self, value: str) -> None:
        self._logs().append(value)
    def clear(self) -> None:
        self._logs().clear()
    def __iter__(self) -> Iterator[str]:
        return iter(list(self._logs()))
    def __len__(self) -> int:
        return len(self._logs())
    def __bool__(self) -> bool:
        return bool(self._logs())
    def __getitem__(self, item):
        return self._logs()[item]
    def __delitem__(self, item) -> None:
        del self._logs()[item]


class _SessionFormProxy:
    def _form(self) -> dict[str, str]:
        return _state()["last_form"]
    def __getitem__(self, key: str) -> str:
        return self._form()[key]
    def __setitem__(self, key: str, value: str) -> None:
        self._form()[key] = value
    def get(self, key: str, default: str = "") -> str:
        return self._form().get(key, default)
    def keys(self):
        return list(self._form().keys())
    def items(self):
        return self._form().items()
    def values(self):
        return self._form().values()
    def __iter__(self):
        return iter(self._form())
    def __len__(self) -> int:
        return len(self._form())


logs = _SessionLogsProxy()
last_form = _SessionFormProxy()

PROVIDER_MODELS: dict[str, list[str]] = {
    "google": [
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview (thinking_level=high)",
    ],
    "openai": ["gpt-5.4"],
    "deepseek": ["deepseek-v4-pro"],
    "anthropic": ["claude-sonnet-4-6"],
    "together": ["meta-llama/Llama-3.3-70B-Instruct-Turbo"],
}


MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini:gemini-3.1-pro-preview": {
        "OUTPUT_USD_PER_1M_TOKENS": 12.0,
        "INPUT_USD_PER_1M_TOKENS": 2.0,
        "CACHE_BUILD_USD_PER_1M_TOKENS": 0.2,
        "CACHE_STORAGE_USD_PER_1M_TOKENS": 4.5,
    },
    "gemini:gemini-3-flash-preview": {
        "OUTPUT_USD_PER_1M_TOKENS": 3.9,
        "INPUT_USD_PER_1M_TOKENS": 0.5,
        "CACHE_BUILD_USD_PER_1M_TOKENS": 0.05,
        "CACHE_STORAGE_USD_PER_1M_TOKENS": 1.0,
    },
    "gemini:gemini-3.1-flash-lite-preview": {
        "OUTPUT_USD_PER_1M_TOKENS": 1.95,
        "INPUT_USD_PER_1M_TOKENS": 0.25,
        "CACHE_BUILD_USD_PER_1M_TOKENS": 0.025,
        "CACHE_STORAGE_USD_PER_1M_TOKENS": 1.0,
    },
    "openai:gpt-5.4-thinking": {
        "OUTPUT_USD_PER_1M_TOKENS": 15.0,
        "INPUT_USD_PER_1M_TOKENS": 2.5,
        "CACHE_BUILD_USD_PER_1M_TOKENS": 0.25,
        "CACHE_STORAGE_USD_PER_1M_TOKENS": 0.0,
    },
    "openai:gpt-5.4": {
        "OUTPUT_USD_PER_1M_TOKENS": 15.0,
        "INPUT_USD_PER_1M_TOKENS": 2.5,
        "CACHE_BUILD_USD_PER_1M_TOKENS": 0.25,
        "CACHE_STORAGE_USD_PER_1M_TOKENS": 0.0,
    },
    "together:meta-llama/Llama-3.3-70B-Instruct-Turbo": {
        "OUTPUT_USD_PER_1M_TOKENS": 0.88,
        "INPUT_USD_PER_1M_TOKENS": 0.88,
        "CACHE_BUILD_USD_PER_1M_TOKENS": 0.88,
        "CACHE_STORAGE_USD_PER_1M_TOKENS": 0.0,
    },
    "together:menagedreef_265f/meta-llama/Llama-3.3-70B-Instruct-Turbo-5f2c0da6": {
        "OUTPUT_USD_PER_1M_TOKENS": 0.88,
        "INPUT_USD_PER_1M_TOKENS": 0.88,
        "CACHE_BUILD_USD_PER_1M_TOKENS": 0.88,
        "CACHE_STORAGE_USD_PER_1M_TOKENS": 0.0,
    },
    "deepseek:deepseek-v4-pro": {
        "OUTPUT_USD_PER_1M_TOKENS": 0.87,
        "INPUT_USD_PER_1M_TOKENS": 0.435,
        "CACHE_BUILD_USD_PER_1M_TOKENS": 0.028,
        "CACHE_STORAGE_USD_PER_1M_TOKENS": 0.0,
    },
    "deepseek:deepseek-reasoner": {
        "OUTPUT_USD_PER_1M_TOKENS": 0.42,
        "INPUT_USD_PER_1M_TOKENS": 0.28,
        "CACHE_BUILD_USD_PER_1M_TOKENS": 0.028,
        "CACHE_STORAGE_USD_PER_1M_TOKENS": 0.0,
    },
}

DOUBLE_COUNTING_OUTPUT_MODELS = {"openai:gpt-5.2-thinking"}

COUNTEREXAMPLE_MODES = ["deterministic short counterexample", "minimal counterexample"]


def _clean_value(value: str) -> str:
    value = (value or "").strip()
    if value.endswith(".0") and value[:-2].isdigit():
        return value[:-2]
    return value


def _read_unique_from_results(column: str, *, transform=lambda x: x) -> list[str]:
    out: list[str] = []
    seen = set()
    for path in [DEFAULT_RESULTS_CSV, FALLBACK_RESULTS_CSV]:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    val = transform(_clean_value(str(row.get(column, ""))))
                    if val and val.lower() != "nan" and val not in seen:
                        seen.add(val)
                        out.append(val)
        except Exception:
            pass
    return out


def _merge_options(defaults: Iterable[str], table_values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for value in list(defaults) + list(table_values):
        value = _clean_value(str(value))
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _append_log(line: str) -> None:
    logs.append(line)
    if len(logs) > 25000:
        del logs[:5000]


def _normalize_model_name(model_name: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", (model_name or "").strip())


def _is_flash_lite_model(provider: str, model_name: str) -> bool:
    return (provider or "").strip().lower() == "google" and _normalize_model_name(model_name) == FLASH_LITE_MODEL


def _csv_path_for_session(sid: str) -> Path:
    form = _state(sid)["last_form"]
    output_dir = ROOT / form.get("output_dir", _session_output_dir(sid))
    csv_name = form.get("experiment_csv", "results.csv") or "results.csv"
    return output_dir / csv_name


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists() or not path.is_file():
        return [], []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return [], []
            return list(reader.fieldnames), list(reader)
    except Exception:
        return [], []


def _append_rows_to_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if not fieldnames or not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _append_rows_to_google_sheet(sheet_name: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Append rows to Google Sheets when GOOGLE_SHEET_ID and credentials are configured.

    Render env vars supported:
      GOOGLE_SHEET_ID
      GOOGLE_SHEETS_CREDENTIALS_JSON  (service account JSON as one line)
      GOOGLE_APPLICATION_CREDENTIALS  (path to service account JSON file)
    """
    if not fieldnames or not rows:
        return
    spreadsheet_id = os.environ.get("GOOGLE_SHEET_ID", "").strip()
    if not spreadsheet_id:
        return
    try:
        import gspread
        creds_json = os.environ.get("GOOGLE_SHEETS_CREDENTIALS_JSON", "").strip()
        if creds_json:
            credentials = json.loads(creds_json)
            gc = gspread.service_account_from_dict(credentials)
        else:
            gc = gspread.service_account()

        sh = gc.open_by_key(spreadsheet_id)
        try:
            ws = sh.worksheet(sheet_name)
        except Exception:
            ws = sh.add_worksheet(title=sheet_name, rows=1000, cols=max(20, len(fieldnames)))
            ws.append_row(fieldnames, value_input_option="RAW")

        existing_header = ws.row_values(1)
        if not existing_header:
            ws.append_row(fieldnames, value_input_option="RAW")
        elif existing_header != fieldnames:
            # Preserve old columns and append any new columns at the end.
            merged = list(existing_header)
            for name in fieldnames:
                if name not in merged:
                    merged.append(name)
            if merged != existing_header:
                ws.update("1:1", [merged])
            fieldnames = merged

        values = [[str(row.get(col, "")) for col in fieldnames] for row in rows]
        ws.append_rows(values, value_input_option="RAW")
    except Exception as exc:
        _append_log(f"Google Sheets append failed ({sheet_name}): {type(exc).__name__}: {exc}")


def _latest_cost_value_from_logs(text: str) -> float:
    metrics = _latest_token_metrics_from_logs(text)
    value = metrics.get("cost_so_far_usd") if isinstance(metrics, dict) else None
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _finalize_run_outputs(sid: str) -> None:
    state = _state(sid)
    if state.get("finalized_once"):
        return
    state["finalized_once"] = True

    text = "\n".join(state.get("logs", []))
    result = _run_result()
    ended_at = datetime.now(timezone.utc).isoformat()
    form = dict(state.get("last_form", {}))
    session_csv = _csv_path_for_session(sid)
    fieldnames, rows = _read_csv_rows(session_csv)

    enriched_rows: list[dict[str, Any]] = []
    if rows and fieldnames:
        extra_fields = [
            "launcher_session_id",
            "launcher_result",
            "launcher_ended_at_utc",
            "launcher_auto_key_used",
            "launcher_final_cost_usd",
        ]
        merged_fields = list(fieldnames)
        for col in extra_fields:
            if col not in merged_fields:
                merged_fields.append(col)
        cost = _latest_cost_value_from_logs(text)
        for row in rows:
            out = dict(row)
            out.update({
                "launcher_session_id": sid,
                "launcher_result": result,
                "launcher_ended_at_utc": ended_at,
                "launcher_auto_key_used": "1" if state.get("auto_key_used") else "0",
                "launcher_final_cost_usd": f"{cost:.8f}",
            })
            enriched_rows.append(out)
        _append_rows_to_csv(GLOBAL_RESULTS_CSV, merged_fields, enriched_rows)
        _append_rows_to_google_sheet("all_users_results", merged_fields, enriched_rows)

    if state.get("auto_key_used"):
        cost = _latest_cost_value_from_logs(text)
        day = datetime.now().strftime("%Y-%m-%d")
        cost_fields = ["date", "session_id", "ended_at_utc", "provider", "model_name", "result", "cost_usd"]
        cost_row = {
            "date": day,
            "session_id": sid,
            "ended_at_utc": ended_at,
            "provider": form.get("api_provider", ""),
            "model_name": form.get("model_name", ""),
            "result": result,
            "cost_usd": f"{cost:.8f}",
        }
        _append_rows_to_csv(AUTO_KEY_DAILY_COSTS_CSV, cost_fields, [cost_row])
        _append_rows_to_google_sheet("auto_key_daily_costs", cost_fields, [cost_row])


def _session_html_files(sid: str) -> list[Path]:
    out_dir = (ROOT / _session_output_dir(sid)).resolve()
    if not out_dir.exists():
        return []
    return sorted([p for p in out_dir.rglob("*.html") if p.is_file()])


def _zip_relative_path_for_session_file(path: Path, out_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(out_dir)).replace("\\", "/")
    except Exception:
        return path.name


def _localize_path_for_results_zip(value: str, sid: str, out_dir: Path) -> str:
    """Convert server/local artifact URLs in downloaded results to paths inside the ZIP.

    The app itself still uses Render URLs while running. Only the downloaded ZIP
    copy is rewritten so result rows point to files that exist inside the ZIP,
    e.g. html/session_...html instead of file:////opt/render/.../html/session_...html.
    """
    if not value:
        return value

    text = str(value)

    def decode_html_artifact(match: re.Match[str]) -> str:
        raw = unquote(match.group(1))
        return _localize_path_for_results_zip(raw, sid, out_dir)

    text = re.sub(r'/html_artifact\?[^\s,;"\'<>]*?path=([^&\s,;"\'<>]+)[^\s,;"\'<>]*', decode_html_artifact, text)

    prefixes = []
    out_posix = out_dir.as_posix()
    out_str = str(out_dir)
    prefixes.extend([
        out_posix + "/",
        out_str + os.sep,
        "file:///" + out_posix.lstrip("/") + "/",
        "file://" + out_posix + "/",
        "file:///" + out_str.replace("\\", "/").lstrip("/") + "/",
    ])

    session_fragment = f"runs/sessions/{sid}/"
    generic_session_fragment = r"runs[/\\]sessions[/\\][^/\\]+[/\\]"
    for prefix in prefixes:
        text = text.replace(prefix, "")

    # Fallback for Render/Linux absolute paths, Windows paths, or any file URL
    # containing a session output directory. Keep only the path inside it.
    text = re.sub(rf'file:/*[^\s,;"\'<>]*?{re.escape(session_fragment)}', '', text)
    text = re.sub(rf'file:/*[^\s,;"\'<>]*?{generic_session_fragment}', '', text)
    text = re.sub(rf'[A-Za-z]:[/\\][^\s,;"\'<>]*?{generic_session_fragment}', '', text)
    text = re.sub(rf'/[^\s,;"\'<>]*?{re.escape(session_fragment)}', '', text)
    text = re.sub(rf'/[^\s,;"\'<>]*?{generic_session_fragment}', '', text)

    artifact_dirs = "html|DFA|evaluations|language_similarity_details|L_star_comparisons|TTT_comparisons|graphs\.pdf|launcher_logs\.txt|graph_generation_log\.txt|results\.csv"
    text = re.sub(rf"file:/*(?=({artifact_dirs}))", "", text)
    text = re.sub(r"(?P<prefix>['\"=(:\s])html/(?P<dir>DFA|evaluations|language_similarity_details|L_star_comparisons|TTT_comparisons)/", r"\g<prefix>\g<dir>/", text)
    if text.startswith("html/DFA/"):
        text = text[len("html/"):]
    if text.startswith("html/evaluations/"):
        text = text[len("html/"):]
    if text.startswith("html/language_similarity_details/"):
        text = text[len("html/"):]
    return text.replace("\\", "/")


def _localized_results_csv_text_for_zip(session_csv: Path, sid: str, out_dir: Path) -> str:
    if not session_csv.exists():
        return ""
    raw = session_csv.read_text(encoding="utf-8-sig", errors="replace")
    try:
        rows = list(csv.DictReader(raw.splitlines()))
        fieldnames = csv.DictReader(raw.splitlines()).fieldnames
        if not fieldnames:
            return raw
        import io
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            localized_row = {}
            for k in fieldnames:
                localized_value = _localize_path_for_results_zip(str(row.get(k, "")), sid, out_dir)
                if k == "conversation_link" and localized_value and not localized_value.startswith("="):
                    # CSV files cannot know the absolute folder chosen by the user
                    # after extraction, so use a relative spreadsheet hyperlink.
                    # When the CSV is opened from the extracted ZIP root, this
                    # points to html/session_....html regardless of the folder name.
                    safe_value = localized_value.replace('"', '""')
                    localized_value = f'=HYPERLINK("{safe_value}","{safe_value}")'
                localized_row[k] = localized_value
            writer.writerow(localized_row)
        return buf.getvalue()
    except Exception:
        return _localize_path_for_results_zip(raw, sid, out_dir)


def _localized_html_content_for_zip(path: Path, sid: str, out_dir: Path) -> str:
    content = path.read_text(encoding="utf-8", errors="replace")
    content = _localize_path_for_results_zip(content, sid, out_dir)

    # Files such as html/session_....html live one directory below the ZIP root.
    # Links from them to ZIP-root folders must therefore go up one level.
    try:
        rel = path.resolve().relative_to(out_dir).as_posix()
    except Exception:
        rel = path.name
    if rel.startswith("html/"):
        root_dirs = "DFA|evaluations|language_similarity_details|L_star_comparisons|TTT_comparisons"
        content = re.sub(
            rf"(?P<prefix>href=[\"']|src=[\"']|url\(|[\"'=:\s(])(?P<dir>{root_dirs})/",
            lambda m: f"{m.group('prefix')}../{m.group('dir')}/",
            content,
        )
        content = re.sub(
            rf"(?P<prefix>href=[\"']|src=[\"']|url\(|[\"'=:\s(])html/(?P<dir>{root_dirs})/",
            lambda m: f"{m.group('prefix')}../{m.group('dir')}/",
            content,
        )
    return content


def _run_graph_generation_for_zip(session_csv: Path, sid: str, out_dir: Path) -> tuple[Path | None, str]:
    script = ROOT / "create_graphs.py"
    if not script.exists():
        return None, "create_graphs.py was not found in the project root, so graphs were not generated."
    if not session_csv.exists():
        return None, "results.csv was not found, so graphs were not generated."

    localized_csv = out_dir / "results_for_graphs.csv"
    graphs_pdf = out_dir / "graphs.pdf"
    try:
        localized_csv.write_text(_localized_results_csv_text_for_zip(session_csv, sid, out_dir), encoding="utf-8", newline="")
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        proc = subprocess.run(
            [sys.executable, str(script), str(localized_csv), "--output-pdf", str(graphs_pdf)],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=env,
            check=False,
        )
        log = proc.stdout or ""
        if proc.returncode == 0 and graphs_pdf.exists():
            return graphs_pdf, log or "Graphs generated successfully."
        return None, log or f"create_graphs.py exited with code {proc.returncode}."
    except Exception as exc:
        return None, f"Graph generation failed: {type(exc).__name__}: {exc}"


def _make_results_zip(sid: str) -> Path:
    state = _state(sid)
    out_dir = (ROOT / _session_output_dir(sid)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"session_results_{sid}.zip"
    session_csv = _csv_path_for_session(sid)
    graph_pdf, graph_log = _run_graph_generation_for_zip(session_csv, sid, out_dir)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if session_csv.exists():
            zf.writestr("results.csv", _localized_results_csv_text_for_zip(session_csv, sid, out_dir))
        for html_path in _session_html_files(sid):
            if html_path == zip_path:
                continue
            arcname = _zip_relative_path_for_session_file(html_path, out_dir)
            try:
                zf.writestr(arcname, _localized_html_content_for_zip(html_path, sid, out_dir))
            except Exception:
                zf.write(html_path, arcname=arcname)
        if graph_pdf and graph_pdf.exists():
            zf.write(graph_pdf, arcname="graphs.pdf")
    return zip_path


def _reader_thread(pipe) -> None:
    try:
        for line in iter(pipe.readline, ""):
            _append_log(line.rstrip("\n"))
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _make_target_preview_html() -> str:
    """Draw exactly the same hidden target DFA that main.py will create."""
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        from output_paths import set_output_dir
        from dfa_factory import make_random_dfa
        from dfa_class import MinimalDFA
        from utils import get_counterexample_mode, get_minimal_counterexample

        output_dir = last_form.get("output_dir", "runs") or "runs"
        set_output_dir(output_dir)

        raw_dfa = make_random_dfa(
            n_states=int(last_form["n_states"]),
            alphabet_size=int(last_form["alphabet_size"]),
            seed=int(last_form["seed"]),
        )

        mdfa = MinimalDFA.from_dfa(
            raw_dfa,
            run_strategy=True,
            minimal_counterexample=get_minimal_counterexample(last_form["counterexample_mode"]),
            counterexample_max_extra_len=3,
            counterexample_mode=get_counterexample_mode(last_form["counterexample_mode"]),
        )
        drawn = mdfa.draw()
        if drawn:
            return str(drawn)
    except Exception as exc:
        _append_log(f"Target preview failed: {type(exc).__name__}: {exc}")
    return ""



def _write_target_dfa_html() -> str:
    '''Create a standalone local HTML file for the hidden target DFA.'''
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        from output_paths import set_output_dir, get_artifact_dir
        from dfa_factory import make_random_dfa
        from dfa_class import MinimalDFA
        from utils import get_counterexample_mode, get_minimal_counterexample

        output_dir = last_form.get("output_dir", "runs") or "runs"
        set_output_dir(output_dir)

        raw_dfa = make_random_dfa(
            n_states=int(last_form["n_states"]),
            alphabet_size=int(last_form["alphabet_size"]),
            seed=int(last_form["seed"]),
        )

        dfa = MinimalDFA.from_dfa(
            raw_dfa,
            run_strategy=True,
            minimal_counterexample=get_minimal_counterexample(last_form["counterexample_mode"]),
            counterexample_max_extra_len=3,
            counterexample_mode=get_counterexample_mode(last_form["counterexample_mode"]),
        )

        out_dir = get_artifact_dir("DFA")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (
            f"target_DFA_states_{last_form['n_states']}_alphabet_{last_form['alphabet_size']}_seed_{last_form['seed']}.html"
        )

        states = sorted([str(s) for s in dfa.states])
        finals = {str(s) for s in dfa.final_states}
        initial = str(dfa.initial_state)
        alphabet = sorted([str(a) for a in dfa.input_symbols])

        import math as _math
        n = max(1, len(states))
        cx, cy, radius = 420, 330, 210
        pos = {}
        for i, state in enumerate(states):
            angle = -_math.pi / 2 + 2 * _math.pi * i / n
            pos[state] = (cx + radius * _math.cos(angle), cy + radius * _math.sin(angle))

        def esc(x: object) -> str:
            return html_lib.escape(str(x), quote=True)

        edge_groups: dict[tuple[str, str], list[str]] = {}
        for src, trans in dfa.transitions.items():
            for sym, dst in trans.items():
                edge_groups.setdefault((str(src), str(dst)), []).append(str(sym))

        svg_parts = []
        svg_parts.append('<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#344054" /></marker></defs>')

        for (src, dst), syms in edge_groups.items():
            x1, y1 = pos[src]
            x2, y2 = pos[dst]
            label = ",".join(sorted(syms))
            if src == dst:
                svg_parts.append(
                    f'<path d="M {x1-35:.1f} {y1-35:.1f} C {x1-105:.1f} {y1-120:.1f}, {x1+105:.1f} {y1-120:.1f}, {x1+35:.1f} {y1-35:.1f}" '
                    f'fill="none" stroke="#344054" stroke-width="2.2" marker-end="url(#arrow)"/>'
                )
                svg_parts.append(f'<text x="{x1:.1f}" y="{y1-105:.1f}" text-anchor="middle" class="edge-label">{esc(label)}</text>')
            else:
                dx, dy = x2 - x1, y2 - y1
                dist = max(1.0, (dx*dx + dy*dy) ** 0.5)
                r_node = 36
                sx = x1 + dx / dist * r_node
                sy = y1 + dy / dist * r_node
                ex = x2 - dx / dist * r_node
                ey = y2 - dy / dist * r_node
                mx, my = (sx + ex) / 2, (sy + ey) / 2
                svg_parts.append(
                    f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
                    f'stroke="#344054" stroke-width="2.2" marker-end="url(#arrow)"/>'
                )
                svg_parts.append(f'<text x="{mx:.1f}" y="{my-8:.1f}" text-anchor="middle" class="edge-label">{esc(label)}</text>')

        for i, state in enumerate(states):
            x, y = pos[state]
            if state == initial and state in finals:
                fill = "#f1c40f"
            elif state == initial:
                fill = "#22c55e"
            elif state in finals:
                fill = "#ef4444"
            else:
                fill = "#4f46e5"
            svg_parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="36" fill="{fill}" stroke="#101828" stroke-width="2"/>')
            if state in finals:
                svg_parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="29" fill="none" stroke="#ffffff" stroke-width="3"/>')
            svg_parts.append(f'<text x="{x:.1f}" y="{y+7:.1f}" text-anchor="middle" class="node-label">q{i}</text>')
            svg_parts.append(f'<title>{esc(state)}</title>')

        legend = f"Initial: green · Final: red/double circle · Alphabet: {esc(', '.join(alphabet))}"
        regex_note = ""
        if last_form.get("target_source") == "regex" and last_form.get("regex"):
            regex_note = f'<span class="target-regex">({esc(last_form.get("regex", ""))})</span>'
        html_doc = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Hidden Target DFA</title>
<style>
  body {{ margin:0; font-family:Arial, sans-serif; background:#ffffff; color:#172033; }}
  .wrap {{ padding:12px; }}
  h3 {{ margin:0 0 4px; font-size:16px; }}
  .target-regex {{ font-size:12px; font-weight:600; color:#667085; margin-left:6px; vertical-align:middle; }}
  .meta {{ color:#667085; font-size:12px; margin-bottom:8px; }}
  svg {{ width:100%; height:100%; min-height:300px; border:1px solid #e5e7eb; border-radius:12px; background:#ffffff; display:block; }}
  .node-label {{ fill:#ffffff; font-size:18px; font-weight:700; }}
  .edge-label {{ fill:#111827; font-size:15px; font-weight:700; paint-order:stroke; stroke:#ffffff; stroke-width:4px; stroke-linejoin:round; }}
</style>
</head>
<body>
<div class="wrap">
  <h3>Hidden Target DFA{regex_note}</h3>
  <div class="meta">{legend}</div>
  <svg viewBox="-70 -55 980 780" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">{''.join(svg_parts)}</svg>
</div>
</body>
</html>'''
        out_path.write_text(html_doc, encoding="utf-8")
        _append_log(f"Target DFA HTML: {out_path}")
        return str(out_path)
    except Exception as exc:
        _append_log(f"Target DFA HTML failed: {type(exc).__name__}: {exc}")
        return ""



def _popen_kwargs_for_stoppable_process() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _request_process_stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return

    if os.name == "nt":
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            return
        except Exception:
            pass

    try:
        proc.terminate()
    except Exception:
        pass


def _kill_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return
        except Exception:
            pass

    try:
        os.killpg(proc.pid, signal.SIGKILL)
        return
    except Exception:
        pass

    try:
        proc.kill()
    except Exception:
        pass

def _run_command(cmd: list[str], sid: str) -> None:
    token = _current_sid.set(sid)
    state = _state(sid)
    state["running"] = True
    if not state["logs"]:
        _append_log("BUDGET_WAIT::Running L* and TTT to compute the query budget for the LLM")
    state["current_full_report_path"] = ""
    state["cached_full_report_path"] = ""
    _append_log("Launcher started.")

    safe_cmd = []
    skip_next = False
    for i, x in enumerate(cmd):
        if skip_next:
            skip_next = False
            continue
        if x == "--api-key" and i + 1 < len(cmd):
            safe_cmd += [x, "***"]
            skip_next = True
        else:
            safe_cmd.append("***" if "sk-" in x else x)
    _append_log("Command: " + " ".join(shlex.quote(x) for x in safe_cmd))

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["LAUNCHER_SESSION_ID"] = sid
    env["STOP_REQUEST_FLAG_PATH"] = str(state["stop_flag_path"])

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            **_popen_kwargs_for_stoppable_process(),
        )
        state["process"] = proc
        if proc.stdout is not None:
            _reader_thread(proc.stdout)
        code = proc.wait()
        _append_log(f"Launcher finished with exit code {code}.")
    except Exception as exc:
        _append_log(f"Launcher error: {type(exc).__name__}: {exc}")
    finally:
        state["running"] = False
        state["process"] = None
        try:
            _finalize_run_outputs(sid)
        except Exception as exc:
            _append_log(f"Launcher finalize error: {type(exc).__name__}: {exc}")
        _current_sid.reset(token)

def _extract_tool_json_blocks(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in re.finditer(r"<TOOL_RESULT>\s*(\{.*?\})\s*</TOOL_RESULT>", text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def _format_word(value: Any) -> str:
    """Display automata words compactly: no spaces between letters; empty word as ε."""
    if value is None:
        raw = ""
    elif isinstance(value, (list, tuple)):
        raw = "".join(str(x) for x in value)
    else:
        raw = str(value)

    raw = raw.strip()
    if raw in {"", "ε", "epsilon", "EPSILON", "Epsilon"}:
        return "ε"

    # Some logs print words as characters separated by spaces, e.g. "a b a".
    # For this UI we want the word itself, e.g. "aba".
    return "".join(raw.split())


def _candidate_to_text(candidate: Any) -> str:
    if isinstance(candidate, dict):
        states = candidate.get("states", [])
        start = candidate.get("start_state", "")
        accept = candidate.get("accept_states", [])
        transitions = candidate.get("transitions", [])
        return (
            f"states={states}\n"
            f"start={start}\n"
            f"accept={accept}\n"
            f"transitions={transitions}"
        )
    if candidate is None:
        return "candidate DFA was submitted"
    return str(candidate)




def _analysis_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    analysis = payload.get("noninformative_analysis") if isinstance(payload, dict) else None
    if not isinstance(analysis, dict):
        return {}

    is_noninformative = bool(analysis.get("is_noninformative"))
    if not is_noninformative:
        return {}

    kind = str(analysis.get("kind") or "").strip()
    details = str(analysis.get("details") or "").strip()

    return {
        "is_noninformative": "1",
        "text": "Non-informative",
        "kind": kind,
        "details": details,
    }


def _passive_learning_by_call_from_logs(text: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for m in re.finditer(
        r"PASSIVE_LEARNING_ANALYSIS::CALL=([^:\n\r]+)::JSON=(\{.*?\})(?=\n|\r|$)",
        text,
        flags=re.IGNORECASE,
    ):
        call = m.group(1).strip()
        try:
            payload = json.loads(m.group(2))
        except Exception:
            continue
        if isinstance(payload, dict):
            out[call] = payload
    return out


def _language_similarity_by_call_from_logs(text: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for m in re.finditer(
        r"LANGUAGE_SIMILARITY_ANALYSIS::CALL=([^:\n\r]+)::JSON=(\{.*?\})(?=\n|\r|$)",
        text,
        flags=re.IGNORECASE,
    ):
        call = m.group(1).strip()
        try:
            payload = json.loads(m.group(2))
        except Exception:
            continue
        if isinstance(payload, dict):
            out[call] = payload
    return out

def _token_metrics_by_step_from_logs(text: str) -> dict[str, dict[str, Any]]:
    """Map model step/tool-call number -> token metrics for launcher UI.

    Parsed lines are printed by model_router.py after every model response:
      TOKEN_METRICS::STEP=3::JSON={...}

    The UI shows per-step input tokens, cached input tokens, and cumulative
    output tokens so viewers can watch token usage grow during the game.
    """
    raw: dict[int, dict[str, Any]] = {}

    for m in re.finditer(
        r"TOKEN_METRICS::STEP=(\d+)::JSON=(\{.*?\})(?=\n|\r|$)",
        text,
        flags=re.IGNORECASE,
    ):
        try:
            step = int(m.group(1))
            payload = json.loads(m.group(2))
        except Exception:
            continue
        if isinstance(payload, dict):
            raw[step] = payload

    out: dict[str, dict[str, Any]] = {}
    output_so_far = 0

    for step in sorted(raw):
        payload = dict(raw[step])
        output_tokens = payload.get("output_tokens")
        if isinstance(output_tokens, int):
            output_so_far += output_tokens
        payload["output_tokens_so_far"] = output_so_far
        out[str(step)] = payload

    return out



def _float_value(obj: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = obj.get(key, default)
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _pricing_model_key(provider: str, model_name: str) -> str:
    """Normalize the selected model to the pricing keys used by graph scripts.

    Inline configuration in parentheses is intentionally ignored, e.g.
    "gemini-3-flash-preview (thinking_level=high)" is priced as
    "gemini:gemini-3-flash-preview".
    """
    provider = (provider or "").strip().lower()
    model = re.sub(r"\s*\([^)]*\)\s*$", "", (model_name or "").strip())
    provider_for_price = "gemini" if provider == "google" else provider
    return f"{provider_for_price}:{model}"


def _format_money(value: float) -> str:
    return f"${value:,.4f}"


def _cost_so_far_from_step_metrics(metrics: dict[str, dict[str, Any]]) -> float | None:
    model_key = _pricing_model_key(last_form.get("api_provider", ""), last_form.get("model_name", ""))
    pricing = MODEL_PRICING.get(model_key)
    if not pricing:
        return None

    input_new_total = 0.0
    cache_total = 0.0
    output_total = 0.0
    last_cache = 0.0

    def step_key(k: str) -> int:
        try:
            return int(float(k))
        except Exception:
            return -1

    for step in sorted(metrics.keys(), key=step_key):
        payload = metrics.get(step) or {}
        input_tokens = _float_value(payload, "input_tokens")
        cache_history_tokens = _float_value(payload, "cache_history_tokens")
        thoughts_tokens = _float_value(payload, "thoughts_tokens")
        output_visible_tokens = _float_value(payload, "output_visible_tokens")
        output_tokens = _float_value(payload, "output_tokens")

        input_new_total += max(input_tokens - cache_history_tokens, 0.0)
        cache_total += cache_history_tokens
        last_cache = cache_history_tokens

        if model_key in DOUBLE_COUNTING_OUTPUT_MODELS:
            effective_output = output_tokens
        else:
            effective_output = thoughts_tokens + output_visible_tokens
            if effective_output <= 0 and output_tokens > 0:
                effective_output = output_tokens
        output_total += effective_output

    output_cost = (output_total / 1_000_000.0) * pricing["OUTPUT_USD_PER_1M_TOKENS"]
    input_cost = (input_new_total / 1_000_000.0) * pricing["INPUT_USD_PER_1M_TOKENS"]
    cache_build_cost = (cache_total / 1_000_000.0) * pricing["CACHE_BUILD_USD_PER_1M_TOKENS"]
    cache_storage_cost = (last_cache / 1_000_000.0) * pricing["CACHE_STORAGE_USD_PER_1M_TOKENS"]
    return output_cost + input_cost + cache_build_cost + cache_storage_cost


def _latest_token_metrics_from_logs(text: str) -> dict[str, Any]:
    """Return the latest token usage summary for the whole run.

    The launcher shows this once under the chat and refreshes it after each
    model step, instead of rendering a token box next to every tool call.
    """
    metrics = _token_metrics_by_step_from_logs(text)
    if not metrics:
        return {}

    def step_key(k: str) -> int:
        try:
            return int(float(k))
        except Exception:
            return -1

    latest_step = max(metrics.keys(), key=step_key)
    payload = dict(metrics.get(latest_step) or {})
    payload["step"] = latest_step

    cost_so_far = _cost_so_far_from_step_metrics(metrics)
    if cost_so_far is not None:
        payload["cost_so_far_usd"] = cost_so_far
        payload["cost_so_far_text"] = _format_money(cost_so_far)

    return payload


def _server_cached_html_path(source_path: str, sid: str) -> str:
    """Copy the final report HTML to a temporary server-side cache and return that path.

    This keeps the analysis iframe available even when the original generated
    artifact path is later moved, localized for ZIP export, or cleaned up.
    """
    if not source_path:
        return ""

    raw = source_path.strip()
    if raw.startswith("file:///"):
        raw = raw[8:]
    elif raw.startswith("file://"):
        raw = raw[7:]

    try:
        src = Path(raw).resolve()
    except Exception:
        return ""

    state = _state(sid)
    cached = str(state.get("cached_full_report_path") or "").strip()
    if cached and Path(cached).exists():
        return cached

    if not src.exists() or not src.is_file():
        if cached and not Path(cached).exists():
            state["cached_full_report_path"] = ""
        return ""

    try:
        cache_dir = ROOT / "runs" / "server_html_cache" / sid
        cache_dir.mkdir(parents=True, exist_ok=True)
        dst = cache_dir / src.name
        shutil.copy2(src, dst)
        state["cached_full_report_path"] = str(dst)
        return str(dst)
    except Exception as exc:
        _append_log(f"Launcher HTML cache error: {type(exc).__name__}: {exc}")
        return str(src)

def _path_to_url(path: str | None, view: str = "full") -> str:
    if not path:
        return ""
    return f"/html_artifact?view={quote(view)}&path={quote(str(path), safe='')}"


def _run_result() -> str:
    txt = "\n".join(logs)
    if "GAME FINISHED - LLM WON" in txt:
        return "won"
    if "GAME FINISHED - LLM LOST" in txt:
        return "lost"
    if "GAME CRASHED" in txt or "Launcher error" in txt or "RUN STOPPED BY USER" in txt:
        return "crashed"
    return "running" if _state().get("running") else "idle"


def _is_query_limit_reached(text: str) -> bool:
    low = text.lower()
    return any(
        phrase in low
        for phrase in [
            "tool_call_limit_reached",
            "tool call limit",
            "query budget",
            "queries exhausted",
            "no queries remaining",
            "remaining_calls=0",
            "max tool calls",
            "game finished - llm lost",
        ]
    )



def _parse_budget_number(value: str) -> int | None:
    try:
        number = float(str(value).strip())
        if number > 0:
            return int(number) if number.is_integer() else int(number)
    except Exception:
        return None
    return None


def _max_tool_calls_from_logs(text: str) -> str:
    """Extract the total tool-call budget from the actual runtime logs.

    Reliable sources seen in the logs:
      TOOL_BUDGET::12
      MODEL INPUT METADATA | remaining_calls=11 | calls_used=1/12 | input chars=178
      You have a total budget of 12 tool calls.
    """
    candidates: list[int] = []
    num = r"(\d+(?:\.\d+)?)"

    def add(value: str) -> None:
        parsed = _parse_budget_number(value)
        if parsed is not None:
            candidates.append(parsed)

    # Best explicit source printed after L*/TTT budget computation.
    for m in re.finditer(rf"TOOL_BUDGET::\s*{num}", text, flags=re.IGNORECASE):
        add(m.group(1))

    # Runtime metadata denominator: calls_used=current/total.
    for m in re.finditer(
        rf"MODEL INPUT METADATA\s*\|.*?calls_used\s*=\s*{num}\s*/\s*{num}",
        text,
        flags=re.IGNORECASE,
    ):
        add(m.group(2))

    # Generic fallback for any calls_used=current/total line.
    for m in re.finditer(rf"calls_used\s*=\s*{num}\s*/\s*{num}", text, flags=re.IGNORECASE):
        add(m.group(2))

    # Prompt/report fallback.
    for pattern in [
        rf"max_tool_calls\s*=\s*{num}",
        rf'"max_tool_calls"\s*:\s*{num}',
        rf"'max_tool_calls'\s*:\s*{num}",
        rf"total budget of\s+{num}\s+tool calls",
        rf"You have a total budget of\s+{num}\s+tool calls",
    ]:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            add(m.group(1))

    return str(max(candidates)) if candidates else ""


def _confirmed_tool_call_numbers_from_logs(text: str) -> set[str]:
    """Return tool-call numbers that were actually sent back to the model.

    The runtime prints valid oracle responses like:
      🔮 Oracle response to tool call #6 sent to model

    Invalid/no-tool outputs are printed like:
      🔮 Oracle response sent to model

    Therefore we only display TOOL_RESULT blocks whose call_count appears in
    an explicit "Oracle response to tool call #N" header. This prevents stale
    or invalid candidate attempts from being shown as real tool calls.

    The first call is also allowed as a fallback for older logs.
    """
    confirmed: set[str] = set()

    for m in re.finditer(
        r"Oracle response to tool call\s*#\s*(\d+)\s+sent to model",
        text,
        flags=re.IGNORECASE,
    ):
        confirmed.add(m.group(1))

    return confirmed


def _metadata_call_numbers_from_logs(text: str) -> set[str]:
    """Backward-compatible alias.

    Older launcher versions used metadata lines to validate calls. The actual
    log format shows that the reliable validation signal is the explicit
    oracle-response header, while MODEL INPUT METADATA is best for extracting
    the total budget.
    """
    return _confirmed_tool_call_numbers_from_logs(text)


def _should_display_tool_call(call: Any, confirmed_calls: set[str]) -> bool:
    """Display a tool call only if the oracle explicitly returned it to the model.

    Exception: call #1 is displayed even if the explicit header is missing,
    for compatibility with older logs.
    """
    try:
        call_s = str(int(float(str(call).strip())))
    except Exception:
        call_s = str(call or "").strip()

    if call_s == "1":
        return True

    return call_s in confirmed_calls


def _tool_call_label(call: Any, total: str) -> str:
    call_s = str(call or "?").strip()
    # Always show "out of" format. If the total is not in the logs yet, show ?.
    return f"#{call_s}/{total if total else '?'}"

def _initial_prompt_from_logs(text: str) -> str:
    """Extract the initial prompt printed by llm_runtime.py, if verbose logs include it."""
    patterns = [
        r"📥\s*Initial prompt sent to model\s*\n!+\s*\n(.*?)\n!{20,}",
        r"Initial prompt sent to model\s*\n!+\s*\n(.*?)\n!{20,}",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
    return ""



def _hypothesis_paths_by_call_from_logs(text: str) -> dict[str, str]:
    """Map EQ call number -> stable hypothesis DFA HTML path.

    The launcher trusts only lines printed by tools.py in this exact format:
      HYPOTHESIS_DFA_LINK::CALL=6::PATH=C:\\...\\DFA_..._6.html

    It intentionally ignores TARGET_DFA_LINK and comparison reports.
    """
    out: dict[str, str] = {}

    for m in re.finditer(
        r"HYPOTHESIS_DFA_LINK::CALL=(\d+)::PATH=([^\n\r]+?\.html)",
        text,
        flags=re.IGNORECASE,
    ):
        call = m.group(1).strip()
        path = m.group(2).strip()

        if "TARGET_DFA_LINK" in path:
            continue

        # First path wins. A later accidental print cannot replace the iframe.
        if call not in out:
            out[call] = path

    return out


def _has_started_agent_tool_call(text: str) -> bool:
    """True as soon as the model has emitted a tool action, even before the oracle response is printed."""
    return bool(re.search(r"^🤖\s*Agent tool call #\d+", text, flags=re.MULTILINE)) or "<TOOL_ACTION>" in text


def _events_from_logs() -> list[dict[str, str]]:
    text = "\n".join(logs)
    events: list[dict[str, str]] = []
    seen = set()
    total_tool_calls = _max_tool_calls_from_logs(text)
    confirmed_calls = _confirmed_tool_call_numbers_from_logs(text)

    initial_prompt = _initial_prompt_from_logs(text)
    if not initial_prompt and (_state().get("running") or "BUDGET_WAIT::" in text):
        events.append({
            "type": "budget_wait",
            "call": "",
            "call_label": "",
            "agent": "",
            "oracle": "Running L* and TTT to compute the query budget for the LLM",
            "oracle_class": "oracle-normal",
            "iframe": "",
            "automaton_text": "",
            "prompt_text": "",
            "analysis": {},
        })

    if initial_prompt:
        events.append({
            "type": "init_prompt",
            "call": "",
            "call_label": "",
            "agent": "",
            "oracle": "Initial prompt sent to model",
            "oracle_class": "oracle-normal",
            "iframe": "",
            "automaton_text": "",
            "prompt_text": initial_prompt,
            "analysis": {},
        })

    hypothesis_paths_by_call = _hypothesis_paths_by_call_from_logs(text)
    passive_learning_by_call = _passive_learning_by_call_from_logs(text)
    language_similarity_by_call = _language_similarity_by_call_from_logs(text)
    token_metrics_by_step = _token_metrics_by_step_from_logs(text)

    for obj in _extract_tool_json_blocks(text):
        tool_outputs = obj.get("tool_outputs")
        if not isinstance(tool_outputs, list) or not tool_outputs:
            continue

        item = tool_outputs[0]
        if not isinstance(item, dict):
            continue

        tool = item.get("tool_name", "")
        call = item.get("call_count", "?")

        if not _should_display_tool_call(call, confirmed_calls):
            continue

        payload = item.get("output") or {}
        err = item.get("error")
        if not isinstance(payload, dict):
            payload = {}

        key = json.dumps([call, tool, payload, err], sort_keys=True, default=str, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)

        if tool == "is_word_in_language":
            word = _format_word(payload.get("word", ""))
            accepted = payload.get("accepted")
            oracle = "True" if accepted is True else "False" if accepted is False else str(accepted or err or "")
            events.append({
                "type": "mq",
                "call": str(call),
                "call_label": _tool_call_label(call, total_tool_calls),
                "agent": f'"{word}"?',
                "oracle": oracle,
                "oracle_class": "oracle-normal",
                "iframe": "",
                "automaton_text": "",
                "analysis": _analysis_from_payload(payload),
                "passive": passive_learning_by_call.get(str(call), {}),
                "similarity": {},
                "token_metrics": token_metrics_by_step.get(str(call), {}),
            })

        elif tool == "evaluate_dfa_candidate":
            candidate = payload.get("_candidate_obj") or payload.get("candidate_dfa")
            optimal = payload.get("optimal")
            raw_witness = payload.get("witness_word", "")
            witness = _format_word(raw_witness)
            report_path = ""

            try:
                call_s = str(int(float(str(call).strip())))
            except Exception:
                call_s = str(call).strip()

            if call_s in hypothesis_paths_by_call:
                report_path = hypothesis_paths_by_call[call_s]

            if optimal is True:
                oracle = "T"
                oracle_class = "oracle-success"
            else:
                oracle = f'counterexample: {witness}' if raw_witness is not None else str(err or "counterexample: not shown")
                oracle_class = "oracle-normal"

            events.append({
                "type": "eq",
                "call": str(call),
                "call_label": _tool_call_label(call, total_tool_calls),
                "agent": "Is this DFA correct?",
                "automaton_text": _candidate_to_text(candidate),
                "oracle": oracle,
                "oracle_class": oracle_class,
                "iframe": _path_to_url(report_path, "candidate"),
                "analysis": _analysis_from_payload(payload),
                "passive": passive_learning_by_call.get(str(call), {}),
                "similarity": language_similarity_by_call.get(str(call), {}),
                "token_metrics": token_metrics_by_step.get(str(call), {}),
            })

    if events and _is_query_limit_reached(text) and _run_result() in {"lost", "crashed"}:
        events[-1]["oracle_class"] = "oracle-fail"

    return events



def _failure_type_from_events(events: list[dict[str, Any]]) -> str:
    """Classify a failed run using the passive learners at the last hypothesis.

    If at least one passive learner could solve from the evidence available
    before the last proposed hypothesis, the LLM had enough information and
    the failure is a reasoning failure. Otherwise it is a planning failure.
    """
    last_eq: dict[str, Any] | None = None
    for ev in events:
        if ev.get("type") == "eq":
            last_eq = ev

    if not last_eq:
        return "planning failure"

    passive = last_eq.get("passive")
    if not isinstance(passive, dict):
        return "planning failure"

    results = passive.get("results")
    if not isinstance(results, list):
        return "planning failure"

    for item in results:
        if isinstance(item, dict) and item.get("success") is True:
            return "reasoning failure"

    return "planning failure"


def _local_file_url(path: str | None) -> str:
    if not path:
        return ""
    return f"/local_file?path={quote(str(path), safe='')}"


def _crash_reason_from_logs(text: str) -> str:
    matches = re.findall(r"^Crash:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if matches:
        return matches[-1].strip()
    if "RUN STOPPED BY USER" in text:
        return "StoppedByUser: the run was stopped by the user"
    m = re.search(r"Launcher error:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return "Unknown error"


def _result_save_note_from_logs(text: str) -> dict[str, str]:
    result = _run_result()
    if result not in {"won", "lost", "crashed"}:
        return {"visible": "", "html": "", "text": ""}

    saved_patterns = [
        r"Game results were saved in row\s*([^:\n\r]+)\s*:\s*([^\n\r]+)",
        r"Game results saved in row\s*([^:\n\r]+)\s*:\s*([^\n\r]+)",
        r"Results were saved in row\s*([^:\n\r]+)\s*:\s*([^\n\r]+)",
        r"Result was saved in row\s*([^:\n\r]+)\s*:\s*([^\n\r]+)",
    ]

    for pattern in saved_patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            row = html_lib.escape(m.group(1).strip(), quote=True)
            path = m.group(2).strip()
            label = html_lib.escape(path, quote=True)
            href = html_lib.escape(_local_file_url(path), quote=True)

            if result == "crashed":
                reason = html_lib.escape(_crash_reason_from_logs(text), quote=True)
                return {
                    "visible": "1",
                    "text": f"A crash row was saved to the results table at {path}. This row contains the context collected so far and should not be counted as a result. Crash: {reason}",
                    "html": (
                        f'A crash row was saved to the results table at '
                        f'<a href="{href}" target="_blank" rel="noopener">{label}</a>. '
                        f'Row: {row}. This row contains the context collected so far and should not be counted as a result.'
                        f'<br>Crash: {reason}'
                    ),
                }

            return {
                "visible": "1",
                "text": f"The game was saved to the results table at {path}.",
                "html": f'The game was saved to the results table at <a href="{href}" target="_blank" rel="noopener">{label}</a>. Row: {row}.'
            }

    if result == "crashed":
        reason = html_lib.escape(_crash_reason_from_logs(text), quote=True)
        return {
            "visible": "1",
            "text": f"The game crashed, but no results-table save confirmation was found. Crash: {reason}",
            "html": f"The game crashed, but no results-table save confirmation was found.<br>Crash: {reason}",
        }

    return {
        "visible": "1",
        "text": "No confirmation was found that this game was saved to the results table.",
        "html": "No confirmation was found that this game was saved to the results table.",
    }

def _target_dfa_path() -> str:
    text = "\n".join(logs)
    m = re.search(r"TARGET_DFA_LINK::([^\n]+\.html)", text)
    return m.group(1).strip() if m else ""


def _latest_hypothesis_path() -> str:
    text = "\n".join(logs)
    matches = re.findall(
        r"HYPOTHESIS_DFA_LINK::CALL=\d+::PATH=([^\n\r]+?\.html)",
        text,
    )
    return matches[-1].strip() if matches else ""

def _latest_comparison_path() -> str:
    for ev in reversed(_events_from_logs()):
        iframe = ev.get("iframe", "")
        m = re.search(r"path=([^&]+)", iframe)
        if m:
            return unquote(html_lib.unescape(m.group(1)))
    return ""


def _first_comparison_path() -> str:
    """Return the first EQ comparison report.

    In each comparison report, iframe #0 is the hidden target DFA and
    iframe #1 is the candidate DFA proposed by the model. The target panel
    should therefore be populated as soon as the first EQ report is created.
    """
    for ev in _events_from_logs():
        iframe = ev.get("iframe", "")
        m = re.search(r"path=([^&]+)", iframe)
        if m:
            return unquote(html_lib.unescape(m.group(1)))
    return ""


def _latest_game_display_path() -> str:
    text = "\n".join(logs)
    patterns = [
        r'Visual game display:\s*(?:click here to view it:)?\s*(file:///[^\s"]+?\.html|[A-Za-z]:[^\n]+?\.html|[^\s]+session_[^\s]+\.html)',
        r'(?:Game display|Full game analysis|Analysis HTML|HTML report):\s*(?:click here to view it:)?\s*(file:///[^\s"]+?\.html|[A-Za-z]:[^\n]+?\.html|[^\s]+session_[^\s]+\.html)',
        r'(file:///[^\s"]*?/html/session_[^\s"]+?\.html|[A-Za-z]:[^\n]*?[/\\]html[/\\]session_[^\n]+?\.html|[^\s]+/html/session_[^\s]+?\.html)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    # Robust fallback: when the log line is missing or was printed in a slightly
    # different format, use the newest session HTML generated under this
    # browser session's output directory.
    try:
        sid = _get_sid()
        out_dir = (ROOT / _session_output_dir(sid)).resolve()
        candidates = sorted((out_dir / "html").glob("session_*.html"), key=lambda x: x.stat().st_mtime)
        if candidates:
            return str(candidates[-1])
    except Exception:
        pass
    return ""



def _is_tool_budget_exhausted(text: str) -> bool:
    """Return True when the logs show that the LLM has no tool calls left."""
    if re.search(r"remaining_calls\s*=\s*0(?:\.0)?\b", text, flags=re.IGNORECASE):
        return True

    num = r"(\d+(?:\.\d+)?)"
    for m in re.finditer(
        rf"calls_used\s*=\s*{num}\s*/\s*{num}",
        text,
        flags=re.IGNORECASE,
    ):
        try:
            used = float(m.group(1))
            total = float(m.group(2))
            if total > 0 and used >= total:
                return True
        except Exception:
            pass

    return False


HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Agentic Automata Learning Runner</title>
<style>
:root{
  --bg:#f3f6fb; --card:#fff; --line:#dbe3ef; --text:#172033; --muted:#667085;
  --agent:#e8f1ff; --oracle:#fff3dc; --accent:#2563eb;
  --win:#dcfce7; --win-border:#22c55e; --lose:#fee2e2; --lose-border:#ef4444;
}
*{box-sizing:border-box}
body{font-family:Arial,sans-serif;background:var(--bg);color:var(--text);margin:14px}
.container{max-width:1260px;margin:auto;display:grid;grid-template-columns:320px 1fr;gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:14px;box-shadow:0 4px 18px rgba(15,23,42,.06)}
h2,h3{margin:0 0 8px}
.target-regex{font-size:12px;font-weight:600;color:#667085;margin-left:6px;vertical-align:middle}
.small{color:var(--muted);font-size:12px;margin:4px 0 10px}
label{display:block;margin:6px 0 3px;font-size:12px;font-weight:700;color:#344054}
input,select{width:100%;padding:6px 8px;border:1px solid #cfd8e3;border-radius:9px;font-size:13px;background:#fff}
.row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
details{margin-top:10px;border:1px solid var(--line);border-radius:12px;background:#fbfdff;overflow:hidden}
summary{cursor:pointer;padding:9px 10px;font-weight:700;font-size:13px}
.details-body{padding:0 10px 10px}
button{width:100%;margin-top:12px;padding:9px 14px;border:0;border-radius:11px;background:var(--accent);color:white;font-weight:700;cursor:pointer}
button.secondary{background:#475467}
button.new-game{background:#16a34a}
button.analysis-btn{background:#7c3aed}
.end-actions{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}
.end-actions button{margin-top:0}
#status{display:inline-block;padding:4px 10px;border-radius:999px;background:#eff8ff;color:#175cd3;font-weight:800;font-size:12px}
.status-won{background:var(--win)!important;color:#166534!important}
.status-lost,.status-crashed{background:var(--lose)!important;color:#991b1b!important}
.mini-frame{width:100%;height:300px;border:1px solid var(--line);border-radius:12px;background:white;display:block;margin:10px auto 0;overflow:hidden}
.full-frame{width:100%;height:82vh;border:1px solid var(--line);border-radius:16px;background:white}
.dfa-legend{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:8px 0 10px;font-size:12px;color:#344054;text-align:left}
.legend-item{display:flex;align-items:center;gap:6px;white-space:nowrap}
.legend-dot{width:12px;height:12px;border-radius:50%;border:1px solid #101828;display:inline-block}
.legend-start{background:#22c55e}
.legend-final{background:#ef4444}
.legend-both{background:#f1c40f}
.legend-none{background:#4f46e5}
.candidate-frame{width:100%;height:300px;border:1px solid rgba(15,23,42,.12);border-radius:12px;background:white;display:block;margin:10px auto 0;overflow:hidden}
.save-note{font-size:11px;line-height:1.35;color:#667085;margin:8px 2px 0;text-align:left}
.save-note a{color:#475467;text-decoration:underline}
.token-usage-footer{font-size:12px;line-height:1.35;color:#475467;background:#f9fafb;border:1px solid #e5e7eb;border-radius:12px;margin:8px 2px 0;padding:8px 10px;font-weight:800;text-align:left}
.token-usage-footer.hidden{display:none!important}
.token-usage-title{font-weight:900;color:#344054}.token-cost-line{margin-top:4px;color:#475467}
.hidden{display:none!important}
.output-card.hidden{display:none!important}
.chat-wrap{height:78vh;overflow:auto;background:linear-gradient(180deg,#f8fbff,#f4f7fb);border:1px solid var(--line);border-radius:16px;padding:18px;scroll-behavior:auto}
.turn{display:flex;flex-direction:column;gap:10px;margin:14px 0}
.turn-tool{display:grid;grid-template-columns:1fr 1fr;gap:10px;align-items:start;margin:14px 0}
.right-stack{display:flex;flex-direction:column;gap:10px;align-self:start}
.msg{width:50%;padding:12px 14px;border-radius:16px;white-space:pre-wrap;font-family:Arial,sans-serif;font-size:18px;font-weight:800;line-height:1.35;border:1px solid rgba(15,23,42,.07)}
.turn-tool .msg{width:100%}
.agent{background:var(--agent);align-self:flex-start;display:flex;flex-direction:column;justify-content:center}
.oracle{background:var(--oracle);align-self:flex-end;text-align:left;min-height:58px;display:flex;flex-direction:column;justify-content:center}
.oracle-success{background:var(--win)!important;border-color:var(--win-border)!important;color:#166534}
.oracle-fail{background:var(--lose)!important;border-color:var(--lose-border)!important;color:#991b1b}
.tool-analysis,.passive-analysis,.similarity-analysis{width:100%;padding:10px 12px;border-radius:14px;background:#fff7ed;border:1px solid #fb923c;font-size:14px;font-weight:800;line-height:1.35;color:#9a3412}
.passive-analysis{background:#f8fafc;border-color:#d0d5dd;color:#344054}
.similarity-analysis{background:#eef6ff;border-color:#93c5fd;color:#1e3a8a}
.analysis-title,.passive-title,.similarity-title{display:flex;align-items:center;gap:7px;font-weight:900}
.analysis-kind,.passive-line,.similarity-line{margin-top:4px;font-size:13px;font-weight:700;color:inherit}
.analysis-details,.passive-words,.similarity-details{margin-top:3px;font-size:12px;font-weight:500;color:#667085;word-break:break-word}
.passive-results{margin-top:8px;white-space:pre-wrap;font-size:13px;font-weight:800;color:#344054}
.bubble-line{display:flex;align-items:center;gap:8px;justify-content:flex-start}
.call-badge{margin-left:auto;font-size:13px;font-weight:900;color:#475467;background:#eef2f7;border:1px solid #d0d5dd;border-radius:999px;padding:3px 8px;white-space:nowrap}
.prompt-card{cursor:pointer;display:block}
.prompt-card summary{list-style:none;cursor:pointer}
.prompt-card summary::-webkit-details-marker{display:none}
.prompt-card > summary.bubble-line{padding:0;margin:0;font-size:18px;font-weight:800;line-height:1.35;color:inherit}
.prompt-card > summary.bubble-line span:not(.emoji){font-size:18px;font-weight:800;line-height:1.35;color:inherit}
.prompt-pre{margin:10px 0 0;background:#fff;border:1px solid rgba(15,23,42,.10);border-radius:10px;padding:10px;white-space:pre-wrap;font-family:Consolas,Monaco,monospace;font-size:12px;font-weight:400;line-height:1.35;color:#172033}
.emoji{font-size:24px;line-height:1;background:transparent}
.typing{background:var(--agent);display:inline-flex;align-items:center;gap:8px;width:auto;max-width:160px}
.dots span{display:inline-block;animation:b 1.2s infinite;font-weight:900;font-size:22px}
.dots span:nth-child(2){animation-delay:.2s}.dots span:nth-child(3){animation-delay:.4s}
@keyframes b{0%,80%,100%{opacity:.2;transform:translateY(0)}40%{opacity:1;transform:translateY(-3px)}}
@media(max-width:930px){.container{grid-template-columns:1fr}.chat-wrap{height:62vh}.msg{width:90%}.turn-tool{grid-template-columns:1fr}.turn-tool .msg,.tool-analysis,.passive-analysis,.similarity-analysis{width:100%}}
</style>
<script>
const providerModels = {{ provider_models_json|safe }};
let wasRunning = false;
let autoScroll = true;
let forceForm = new URLSearchParams(window.location.search).get('new') === '1';
let analysisMode = false;
let lastRenderedEventsKey = "";
let renderLocked = false;
let lockedFrameSrcByCall = {};
let renderedEventKeys = [];

function escapeHtml(s){return String(s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function isFlashLiteSelected(){
  const provider=document.getElementById('api_provider') ? document.getElementById('api_provider').value : '';
  const model=document.getElementById('model_name') ? document.getElementById('model_name').value.replace(/\s*\([^)]*\)\s*$/, '') : '';
  return provider === 'google' && model === 'gemini-3.1-flash-lite-preview';
}
function updateApiKeyVisibility(){
  const apiKeyBox=document.getElementById('api_key_box');
  const apiKeyInput=document.getElementById('api_key');
  if(!apiKeyBox || !apiKeyInput) return;
  const hide = isFlashLiteSelected();
  apiKeyBox.classList.toggle('hidden', hide);
  apiKeyInput.required = !hide;
  if(hide){ apiKeyInput.value=''; }
}
function updateModels(){
  const provider=document.getElementById('api_provider').value;
  const modelSelect=document.getElementById('model_name');
  const current=modelSelect.dataset.initial || modelSelect.value;
  const models=providerModels[provider]||[];
  modelSelect.innerHTML='';
  models.forEach(m=>{const opt=document.createElement('option'); opt.value=m; opt.textContent=m; if(m===current) opt.selected=true; modelSelect.appendChild(opt);});
  modelSelect.dataset.initial='';
  updateApiKeyVisibility();
}
function updateTargetSource(){
  const src=document.getElementById('target_source') ? document.getElementById('target_source').value : 'dataset';
  const regexBox=document.getElementById('regex_box');
  const datasetBox=document.getElementById('dataset_box');
  if(regexBox) regexBox.classList.toggle('hidden', src !== 'regex');
  if(datasetBox) datasetBox.classList.toggle('hidden', src !== 'dataset');
}
function nearBottom(el){ return (el.scrollHeight - el.scrollTop - el.clientHeight) < 80; }
function centerIframeContent(frame){
  function hideScrollbars(){
    try{
      const win = frame.contentWindow;
      const doc = frame.contentDocument || (win ? win.document : null);
      if(!doc) return;
      if(doc.documentElement) doc.documentElement.style.overflow = 'hidden';
      if(doc.body) doc.body.style.overflow = 'hidden';
      const inner = doc.getElementById('inner');
      if(inner){
        inner.setAttribute('scrolling', 'no');
        const innerDoc = inner.contentDocument || (inner.contentWindow ? inner.contentWindow.document : null);
        if(innerDoc){
          if(innerDoc.documentElement) innerDoc.documentElement.style.overflow = 'hidden';
          if(innerDoc.body) innerDoc.body.style.overflow = 'hidden';
        }
      }
    }catch(e){}
  }

  hideScrollbars();
  setTimeout(hideScrollbars, 50);
  setTimeout(hideScrollbars, 200);
  setTimeout(hideScrollbars, 600);

  if(!frame.dataset.centerHandlerAttached){
    frame.addEventListener('load', () => {
      hideScrollbars();
      setTimeout(hideScrollbars, 50);
      setTimeout(hideScrollbars, 200);
      setTimeout(hideScrollbars, 600);
    });
    frame.dataset.centerHandlerAttached = '1';
  }
}
function centerAllDfaFrames(){
  document.querySelectorAll('iframe.candidate-frame, iframe#target_iframe').forEach(centerIframeContent);
}
let latestFullReportUrl = '';
function updateAnalysisButton(data){
  const btn = document.getElementById('analysis_btn');
  if(!btn) return;
  if(data && data.full_report_url){
    latestFullReportUrl = data.full_report_url;
  }
  const ended = data && ['won','lost','crashed','stopped'].includes(data.result);
  const hasReport = !!latestFullReportUrl || !!(data && data.full_report_url);
  btn.classList.toggle('hidden', !(hasReport || ended));
  btn.textContent = analysisMode ? 'Back to run display' : 'Show full game analysis';
}
function showMode(data){
  const params=document.getElementById('params-panel');
  const game=document.getElementById('game-panel');
  const ended = ['won','lost','crashed','stopped'].includes(data.result);
  if(data.running || (ended && !forceForm)){
    params.classList.add('hidden');
    game.classList.remove('hidden');
  } else {
    params.classList.remove('hidden');
    game.classList.add('hidden');
  }
  document.getElementById('new_game_btn').classList.toggle('hidden', !ended);
  const dlBtn = document.getElementById('download_results_btn');
  if(dlBtn){
    dlBtn.classList.toggle('hidden', !ended);
    if(!dlBtn.dataset.downloading){
      dlBtn.textContent = 'Download ZIP of all runs so far';
      dlBtn.title = 'Downloads all games that ran so far in this browser session, not necessarily only the latest game.';
    }
  }
  const stopBtn = document.getElementById('stop_btn');
  if(stopBtn){ stopBtn.classList.toggle('hidden', !data.running); }
  updateAnalysisButton(data);

  const outputCard = document.getElementById('output-card');
  const hasEvents = data.events && data.events.length > 0;
  const shouldShowOutput = data.running || ended || hasEvents || analysisMode;
  if(outputCard){ outputCard.classList.toggle('hidden', !shouldShowOutput || forceForm); }
}
function eventRenderKey(ev){
  return `${ev.type || ''}:${ev.call || ''}:${ev.iframe || ''}`;
}
function renderSingleEvent(ev){
      const cls = ev.oracle_class || 'oracle-normal';
      const callBadge = ev.call_label ? `<span class="call-badge">${escapeHtml(ev.call_label)}</span>` : '';

      if(ev.type === 'budget_wait'){
        return `<div class="turn">
          <div class="msg oracle ${cls}"><div class="bubble-line"><span class="emoji">🔮</span><span>${escapeHtml(ev.oracle)}</span><span class="dots"><span>.</span><span>.</span><span>.</span></span></div></div>
        </div>`;
      }

      if(ev.type === 'init_prompt'){
        return `<div class="turn">
          <details class="msg oracle ${cls} prompt-card">
            <summary class="bubble-line"><span class="emoji">🔮</span><span>${escapeHtml(ev.oracle)}</span></summary>
            <pre class="prompt-pre">${escapeHtml(ev.prompt_text || '')}</pre>
          </details>
        </div>`;
      }


      let frame = '';
      if(ev.iframe){
        const frameKey = `${ev.type || 'event'}:${ev.call || ''}`;
        if(!lockedFrameSrcByCall[frameKey]){
          lockedFrameSrcByCall[frameKey] = ev.iframe;
        }
        const lockedSrc = lockedFrameSrcByCall[frameKey];
        if(lockedSrc){
          frame = `<iframe class="candidate-frame" src="${lockedSrc}" scrolling="no" onload="centerIframeContent(this)"></iframe>`;
        }
      } else if(ev.automaton_text) {
        frame = `<div class="small">${escapeHtml(ev.automaton_text)}</div>`;
      }
      let analysisHtml = '';
      if(ev.analysis && ev.analysis.is_noninformative){
        const kind = ev.analysis.kind ? `<div class="analysis-kind">Type: ${escapeHtml(ev.analysis.kind)}</div>` : '';
        const details = ev.analysis.details ? `<div class="analysis-details">${escapeHtml(ev.analysis.details)}</div>` : '';
        analysisHtml = `<div class="tool-analysis"><div class="analysis-title"><span>${escapeHtml(ev.analysis.text || 'Non-informative')}</span></div>${kind}${details}</div>`;
      }

      let passiveHtml = '';
      if(ev.passive && (ev.passive.accepted_words || ev.passive.rejected_words || ev.passive.results)){
        const fmtWords = (words) => {
          if(!Array.isArray(words) || !words.length) return '{}';
          return `{${words.map(w => escapeHtml(String(w) === '' ? 'ε' : w)).join(', ')}}`;
        };
        const results = Array.isArray(ev.passive.results) ? ev.passive.results.map(r => {
          const alg = r.algorithm || r.key || 'Algorithm';
          const ok = r.success === true ? 'success' : 'failure';
          return `${escapeHtml(alg)}: ${ok}`;
        }).join(' · ') : '';
        passiveHtml = `<div class="passive-analysis">
          <div class="passive-title">Passive learning</div>
          <div class="passive-words">Words in language: ${fmtWords(ev.passive.accepted_words)}</div>
          <div class="passive-words">Words not in language: ${fmtWords(ev.passive.rejected_words)}</div>
          <div class="passive-results">${results}</div>
        </div>`;
      }

      let similarityHtml = '';
      if(ev.type === 'eq' && ev.similarity && (ev.similarity.available !== undefined || ev.similarity.similarity || ev.similarity.error)){
        let sim = 'X';
        const rawSim = ev.similarity.similarity ?? ev.similarity.similarity_float ?? ev.similarity.similarity_short;

        if(rawSim !== undefined && rawSim !== null){

            const rawStr = String(rawSim).trim();

            if(rawStr.toLowerCase().includes('e')){
                sim = "0";
            }
            else if(!isNaN(Number(rawStr))){
                const numericSim = Number(rawStr);

                if(numericSim > 1){
                    sim = "0";
                }
                else{
                    if(rawStr.includes('.')){
                        const parts = rawStr.split('.');
                        sim = `${parts[0]}.${(parts[1] || '').slice(0,3).padEnd(3,'0')}`;
                    }
                    else{
                        sim = `${rawStr}.000`;
                    }
                }
            }
        }

        const detailPath = ev.similarity.detail_path || '';
        const detailUrl = detailPath
          ? `/html_artifact?view=full&path=${encodeURIComponent(detailPath)}`
          : '';

        const comparisonLink = detailUrl
          ? `<div class="similarity-details"><a href="${escapeHtml(detailUrl)}" target="_blank" rel="noopener">Open similarity comparison</a></div>`
          : '';

        const extra = ev.similarity.available === false && ev.similarity.error
          ? `<div class="similarity-details">${escapeHtml(ev.similarity.error)}</div>`
          : '';
        similarityHtml = `<div class="similarity-analysis">
          <div class="similarity-title">Hypothesis similarity: ${escapeHtml(sim)}</div>
          ${comparisonLink}
          ${extra}
        </div>`;
      }

      return `<div class="turn-tool">
        <div class="msg agent"><div class="bubble-line"><span class="emoji">🤖</span><span>${escapeHtml(ev.agent)}</span>${callBadge}</div>${frame}</div>
        <div class="right-stack">
          ${analysisHtml}
          ${passiveHtml}
          ${similarityHtml}
          <div class="msg oracle ${cls}"><div class="bubble-line"><span class="emoji">🔮</span><span>${escapeHtml(ev.oracle)}</span></div></div>
        </div>
      </div>`;

}
function renderEvents(events, isRunning, result, budgetExhausted, toolRequestStarted){
  const host=document.getElementById('chat');
  const full=document.getElementById('full-analysis');
  const shouldStick = autoScroll && nearBottom(host);
  const isTerminal = (result === 'won' || result === 'lost' || result === 'crashed' || result === 'stopped');

  if(analysisMode){
    host.classList.add('hidden');
    full.classList.remove('hidden');
    return;
  }

  full.classList.add('hidden');
  host.classList.remove('hidden');

  if((budgetExhausted || isTerminal) && renderLocked && !analysisMode){
    return;
  }

  events = events || [];
  const hasInitialPrompt = !!(events && events.some(ev => ev.type === 'init_prompt'));
  let parts = events.map(ev => renderSingleEvent(ev));

  const hasToolEvents = events.some(ev => ev.type === 'mq' || ev.type === 'eq');
  const shouldShowLiveThinking = isRunning && hasInitialPrompt && !hasToolEvents && !toolRequestStarted && !budgetExhausted && !isTerminal;
  if(shouldShowLiveThinking){
    parts.push(`<div class="turn"><div class="msg typing"><span class="emoji">🤖</span><span class="dots"><span>.</span><span>.</span><span>.</span></span></div></div>`);
  }

  const currentKeys = events.map(ev => eventRenderKey(ev));
  let canAppendOnly = renderedEventKeys.length <= currentKeys.length;
  for(let i=0; i<renderedEventKeys.length && canAppendOnly; i++){
    if(renderedEventKeys[i] !== currentKeys[i]) canAppendOnly = false;
  }

  const lastEvent = events && events.length ? events[events.length - 1] : null;
  const shouldForceBottom = !!(lastEvent && lastEvent.type === 'eq' && lastEvent.oracle_class === 'oracle-success');

  if(canAppendOnly){
    const typing = host.querySelector('[data-live-typing="1"]');
    if(typing) typing.remove();

    for(let i=renderedEventKeys.length; i<currentKeys.length; i++){
      host.insertAdjacentHTML('beforeend', parts[i] || '');
    }

    if(shouldShowLiveThinking){
      host.insertAdjacentHTML('beforeend', `<div class="turn" data-live-typing="1"><div class="msg typing"><span class="emoji">🤖</span><span class="dots"><span>.</span><span>.</span><span>.</span></span></div></div>`);
    }

    renderedEventKeys = currentKeys.slice();
    centerAllDfaFrames();
  } else {
    host.innerHTML = parts.join('');
    renderedEventKeys = currentKeys.slice();
    centerAllDfaFrames();
  }

  lastRenderedEventsKey = JSON.stringify({
    events: events || [],
    isRunning: isRunning,
    result: result || '',
    budgetExhausted: !!budgetExhausted
  });

  if (shouldStick || shouldForceBottom) {
    host.scrollTop = host.scrollHeight;
  }

  if (budgetExhausted || isTerminal) {
    renderLocked = true;
  }
}

function updateSaveNote(data){
  const note = document.getElementById('save-note');
  if(!note) return;
  if(data.save_note && data.save_note.visible){
    note.innerHTML = data.save_note.html || escapeHtml(data.save_note.text || '');
    note.classList.remove('hidden');
  } else {
    note.innerHTML = '';
    note.classList.add('hidden');
  }
}

function updateTokenUsage(data){
  const box = document.getElementById('token-usage-footer');
  if(!box) return;
  if(analysisMode){
    box.classList.add('hidden');
    return;
  }

  const m = data.token_metrics || {};
  const hasMetrics = m.input_tokens !== undefined || m.output_tokens_so_far !== undefined;
  if(!hasMetrics){
    box.innerHTML = '';
    box.classList.add('hidden');
    return;
  }

  const fmtNum = (v) => (typeof v === 'number' && Number.isFinite(v)) ? v.toLocaleString() : '—';
  const inputTokens = fmtNum(m.input_tokens);
  const cachedTokens = fmtNum(m.cache_history_tokens || 0);
  const outputSoFar = fmtNum(m.output_tokens_so_far);

  const costLine = m.cost_so_far_text
    ? `<div class="token-cost-line">Cost So Far: ${escapeHtml(m.cost_so_far_text)}</div>`
    : '';

  box.innerHTML = `<div><span class="token-usage-title">Token Usage So Far</span> Input: ${inputTokens} (${cachedTokens} cached) Output: ${outputSoFar}</div>${costLine}`;
  box.classList.remove('hidden');
}

function refreshEvents(){
  fetch('/events').then(r=>r.json()).then(data=>{
    showMode(data);
    updateSaveNote(data);
    updateTokenUsage(data);
    const status=document.getElementById('status');
    status.className='';
    if(forceForm && !data.running){ status.textContent='Not in game'; }
    else if(data.result==='won'){ status.textContent='SUCCESS'; status.classList.add('status-won'); autoScroll=false; }
    else if(data.result==='lost'){ status.textContent=(data.failure_type || 'failure').toUpperCase(); status.classList.add('status-lost'); autoScroll=false; }
    else if(data.result==='crashed'){ status.textContent='CRASHED'; status.classList.add('status-crashed'); autoScroll=false; }
    else if(data.result==='stopped'){ status.textContent='STOPPED'; status.classList.add('status-crashed'); autoScroll=false; }
    else if(data.running){ status.textContent='Running'; if(!wasRunning){ autoScroll=true; renderLocked=false; } forceForm=false; analysisMode=false; }
    else status.textContent='Not in game';
    wasRunning = data.running;
    renderEvents(data.events, data.running, data.result, data.budget_exhausted, data.tool_request_started);

    const target=document.getElementById('target_iframe');
    if(data.target_url && !target.dataset.src){
      centerIframeContent(target);
      target.src=data.target_url;
      target.dataset.src=data.target_url;
    }

    const full=document.getElementById('full-analysis');
    if(data.full_report_url){
      latestFullReportUrl = data.full_report_url;
      if(full && full.dataset.src !== data.full_report_url){
        full.src=data.full_report_url;
        full.dataset.src=data.full_report_url;
      }
    }
  });
}
function resetToStartScreenKeepKey(){
  // Reset the launcher UI exactly like starting a fresh game, but keep the
  // form values already stored by Flask, including the API key.
  analysisMode=false;
  latestFullReportUrl='';
  forceForm=true;
  autoScroll=true;
  wasRunning=false;
  lastRenderedEventsKey='';
  renderLocked=false;
  lockedFrameSrcByCall={};
  renderedEventKeys=[];

  window.history.replaceState({},'', '/?new=1');

  const chat=document.getElementById('chat');
  const full=document.getElementById('full-analysis');
  const outputCard=document.getElementById('output-card');
  const target=document.getElementById('target_iframe');
  const status=document.getElementById('status');
  const saveNote=document.getElementById('save-note');
  const tokenUsage=document.getElementById('token-usage-footer');
  const dlBtn=document.getElementById('download_results_btn');

  if(chat){ chat.innerHTML=''; chat.classList.remove('hidden'); }
  if(full){ full.classList.add('hidden'); full.removeAttribute('src'); full.dataset.src=''; }
  if(target){ target.removeAttribute('src'); target.dataset.src=''; }
  if(outputCard){ outputCard.classList.add('hidden'); }
  if(status){ status.textContent='Not in game'; status.className=''; }
  if(saveNote){ saveNote.innerHTML=''; saveNote.classList.add('hidden'); }
  if(tokenUsage){ tokenUsage.innerHTML=''; tokenUsage.classList.add('hidden'); }
  if(dlBtn){ dlBtn.classList.add('hidden'); }
}

async function downloadResultsZip(){
  const btn = document.getElementById('download_results_btn');
  const original = btn ? (btn.textContent || 'Download ZIP of all runs so far') : '';
  if(btn){
    btn.dataset.downloading = '1';
    btn.disabled = true;
    btn.innerHTML = 'Preparing ZIP of all runs so far <span class="dots"><span>.</span><span>.</span><span>.</span></span>';
  }
  try{
    const response = await fetch('/download_results_zip');
    if(!response.ok){
      const msg = await response.text();
      throw new Error(msg || `Download failed with status ${response.status}`);
    }
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'automata_run_results.zip';
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  }catch(e){
    alert(e && e.message ? e.message : 'Failed to download results ZIP');
  }finally{
    if(btn){
      btn.disabled = false;
      delete btn.dataset.downloading;
      btn.textContent = original || 'Download ZIP of all runs so far';
      btn.title = 'Downloads all games that ran so far in this browser session, not necessarily only the latest game.';
    }
  }
}
function stopRun(){
  fetch('/stop',{method:'POST'}).then(() => {
    forceForm=false;
    analysisMode=false;
    renderLocked=false;
    lastRenderedEventsKey='';
    refreshEvents();
  });
}
function newGame(){
  analysisMode=false;
  latestFullReportUrl='';
  forceForm=true;
  autoScroll=true;
  wasRunning=false;
  lastRenderedEventsKey='';
  renderLocked=false;
  lockedFrameSrcByCall={};
  renderedEventKeys=[];
  window.history.replaceState({},'', '/?new=1');
  const chat=document.getElementById('chat');
  const full=document.getElementById('full-analysis');
  const outputCard=document.getElementById('output-card');
  const status=document.getElementById('status');
  const saveNote=document.getElementById('save-note');
  const tokenUsage=document.getElementById('token-usage-footer');
  const dlBtn=document.getElementById('download_results_btn');
  if(chat){ chat.innerHTML=''; chat.classList.remove('hidden'); }
  if(full){ full.classList.add('hidden'); full.removeAttribute('src'); full.dataset.src=''; }
  const target=document.getElementById('target_iframe');
  if(target){ target.removeAttribute('src'); target.dataset.src=''; }
  if(outputCard){ outputCard.classList.add('hidden'); }
  if(status){ status.textContent='Not in game'; status.className=''; }
  if(saveNote){ saveNote.innerHTML=''; saveNote.classList.add('hidden'); }
  if(tokenUsage){ tokenUsage.innerHTML=''; tokenUsage.classList.add('hidden'); }
  if(dlBtn){ dlBtn.classList.add('hidden'); }
  refreshEvents();
}
function showAnalysis(){
  analysisMode = !analysisMode;

  const btn = document.getElementById('analysis_btn');
  const chat = document.getElementById('chat');
  const full = document.getElementById('full-analysis');

  if (analysisMode) {
    if (latestFullReportUrl && full && full.dataset.src !== latestFullReportUrl) {
      full.src = latestFullReportUrl;
      full.dataset.src = latestFullReportUrl;
    }
    if (chat) chat.classList.add('hidden');
    if (full) full.classList.remove('hidden');
    if (btn) btn.textContent = 'Back to run display';
  } else {
    if (full) full.classList.add('hidden');
    if (chat) chat.classList.remove('hidden');
    if (btn) btn.textContent = 'Show full game analysis';

    // Force the next render to refresh the chat view even when the run is
    // already terminal and the event payload did not change.
    lastRenderedEventsKey = '';
  }

  refreshEvents();
}
setInterval(refreshEvents,800);
window.onload=()=>{updateModels();updateApiKeyVisibility();updateTargetSource();refreshEvents();};
</script>
</head>
<body>
<div class="container">
  <div class="card">
    <h2>Agentic Automata Learning Runner</h2>
    <p>Status: <span id="status">Not in game</span></p>

    <div id="params-panel">
      <form method="post" action="/run">
        <div class="row">
          <div><label>API Provider</label><select id="api_provider" name="api_provider" onchange="updateModels()">{% for p in providers %}<option value="{{p}}" {% if form.api_provider==p %}selected{% endif %}>{{p}}</option>{% endfor %}</select></div>
          <div><label>Model</label><select id="model_name" name="model_name" data-initial="{{form.model_name}}" onchange="updateApiKeyVisibility()"></select></div>
        </div>
        <div id="api_key_box"><label>API Key</label><input id="api_key" type="password" name="api_key" autocomplete="off" value="{{form.api_key}}"></div>
        <label>Target Automaton Source</label>
        <select id="target_source" name="target_source" onchange="updateTargetSource()">
          <option value="regex" {% if form.target_source=="regex" %}selected{% endif %}>User regular expression → DFA</option>
          <option value="dataset" {% if form.target_source=="dataset" %}selected{% endif %}>Dataset / random DFA</option>
        </select>
        <div id="regex_box" class="hidden">
          <label>Regular Expression</label>
          <input name="regex" value="{{form.regex}}" placeholder="Example: b*a*">
          <p class="small">Supported: |, *, +, ?, parentheses, implicit concatenation, ε / eps. Alphabet symbols are single characters, e.g. a,b.</p>
        </div>
        <div id="dataset_box">
          <div class="row">
            <div><label>Number of States</label><select name="n_states">{% for n in n_states %}<option value="{{n}}" {% if form.n_states==n %}selected{% endif %}>{{n}}</option>{% endfor %}</select></div>
            <div><label>Seed</label><select name="seed">{% for s in seeds %}<option value="{{s}}" {% if form.seed==s %}selected{% endif %}>{{s}}</option>{% endfor %}</select></div>
          </div>
        </div>
        <details>
          <summary>Experiment options</summary>
          <div class="details-body">
            <div class="row">
              <div><label>Alphabet Size</label><select name="alphabet_size">{% for a in alphabet_sizes %}<option value="{{a}}" {% if form.alphabet_size==a %}selected{% endif %}>{{a}}</option>{% endfor %}</select></div>
              <div><label>Counterexample Mode</label><select name="counterexample_mode">{% for c in counterexample_modes %}<option value="{{c}}" {% if form.counterexample_mode==c %}selected{% endif %}>{{c}}</option>{% endfor %}</select></div>
            </div>
            <label>Algorithm Approximation Ratio</label><select name="algorithm_approximation_ratio">{% for r in ratios %}<option value="{{r}}" {% if form.algorithm_approximation_ratio==r %}selected{% endif %}>{{r}}</option>{% endfor %}</select>
          </div>
        </details>
        <button type="submit">Run</button>
      </form>
    </div>

    <div id="game-panel" class="hidden" style="text-align:center;">
      <h3>Hidden Target DFA{% if form.target_source == "regex" and form.regex %}<span class="target-regex">({{ form.regex }})</span>{% endif %}</h3>
      <div class="dfa-legend" aria-label="DFA color legend">
        <div class="legend-item"><span class="legend-dot legend-start"></span><span>Start state</span></div>
        <div class="legend-item"><span class="legend-dot legend-final"></span><span>Accepting state</span></div>
        <div class="legend-item"><span class="legend-dot legend-both"></span><span>Start + accepting</span></div>
        <div class="legend-item"><span class="legend-dot legend-none"></span><span>Neither</span></div>
      </div>
      <iframe id="target_iframe" class="mini-frame" scrolling="no" onload="centerIframeContent(this)"></iframe>
      <p class="small">You can zoom and drag inside the DFA view to inspect the automaton.</p>
      <button id="stop_btn" type="button" class="secondary hidden" onclick="stopRun()">Stop current run</button>
      <button id="analysis_btn" type="button" class="analysis-btn hidden" onclick="showAnalysis()">Show full game analysis</button>
      <div class="end-actions">
        <button id="download_results_btn" type="button" class="secondary hidden" onclick="downloadResultsZip()" title="Downloads all games that ran so far in this browser session, not necessarily only the latest game.">Download ZIP of all runs so far</button>
        <button id="new_game_btn" type="button" class="new-game hidden" onclick="newGame()">New game</button>
      </div>
    </div>
  </div>
  <div id="output-card" class="card output-card hidden">
    <div id="chat" class="chat-wrap"></div>
    <div id="token-usage-footer" class="token-usage-footer hidden"></div>
    <iframe id="full-analysis" class="full-frame hidden"></iframe>
    <div id="save-note" class="save-note hidden"></div>
  </div>
</div>
</body>
</html>
"""


@app.get("/")
def index():
    alphabet_sizes = [str(i) for i in range(2, 21)]
    ratios = ["1", "1.25", "1.5", "1.75", "2", "2.5", "3"]
    return render_template_string(
        HTML,
        providers=list(PROVIDER_MODELS.keys()),
        provider_models_json=json.dumps(PROVIDER_MODELS, ensure_ascii=False),
        n_states=[str(i) for i in range(1, 21)],
        seeds=[str(i) for i in range(1, 101)],
        alphabet_sizes=alphabet_sizes,
        ratios=ratios,
        counterexample_modes=COUNTEREXAMPLE_MODES,
        form={**dict(_state()["last_form"]), "api_key": "" if _is_flash_lite_model(_state()["last_form"].get("api_provider", ""), _state()["last_form"].get("model_name", "")) else dict(_state()["last_form"]).get("api_key", "")},
    )


@app.post("/run")
def run():
    sid = _get_sid()
    state = _state(sid)
    if state["running"]:
        return "A run is already active in this browser session. Stop it first or wait until it finishes.", 409

    state["running"] = True

    form = state["last_form"]
    for key in list(form.keys()):
        form[key] = request.form.get(key, form[key]).strip()

    form["output_dir"] = _session_output_dir(sid)
    Path(form["output_dir"]).mkdir(parents=True, exist_ok=True)

    state["auto_key_used"] = False
    state["finalized_once"] = False
    if _is_flash_lite_model(form.get("api_provider", ""), form.get("model_name", "")) and not form.get("api_key", "").strip():
        server_google_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if server_google_key:
            form["api_key"] = server_google_key
            state["auto_key_used"] = True
    elif not form.get("api_key", "").strip():
        state["running"] = False
        return "API key is required for this model. The server GOOGLE_API_KEY is used only for gemini-3.1-flash-lite-preview.", 400

    state["current_full_report_path"] = ""
    state["current_target_path"] = ""
    state["logs"].clear()
    _append_log("BUDGET_WAIT::Running L* and TTT to compute the query budget for the LLM")

    try:
        flag_path = state["stop_flag_path"]
        flag_path.parent.mkdir(parents=True, exist_ok=True)
        if flag_path.exists():
            flag_path.unlink()
    except Exception:
        pass

    cmd = [
        sys.executable, "-u", "main.py",
        "--api-provider", form["api_provider"],
        "--model-name", form["model_name"],
        "--api-key", form["api_key"],
        "--n-states", form["n_states"],
        "--seed", form["seed"],
        "--alphabet-size", form["alphabet_size"],
        "--target-source", form["target_source"],
        "--regex", form["regex"],
        "--counterexample-mode", form["counterexample_mode"],
        "--algorithm-approximation-ratio", form["algorithm_approximation_ratio"],
        "--output-dir", form["output_dir"],
        "--experiment-csv", form["experiment_csv"],
    ]
    threading.Thread(target=_run_command, args=(cmd, sid), daemon=True).start()
    return redirect(url_for("index"))

@app.post("/stop")
def stop():
    sid = _get_sid()
    state = _state(sid)
    _append_log("RUN STOPPED BY USER")

    try:
        flag_path = state["stop_flag_path"]
        flag_path.parent.mkdir(parents=True, exist_ok=True)
        flag_path.write_text("stop", encoding="utf-8")
    except Exception as exc:
        _append_log(f"Launcher stop flag error: {type(exc).__name__}: {exc}")

    proc = state.get("process")
    if proc and proc.poll() is None:
        _request_process_stop(proc)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _append_log("Stop did not finish gracefully; killing the running process tree.")
            _kill_process_tree(proc)
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        except Exception as exc:
            _append_log(f"Launcher stop wait error: {type(exc).__name__}: {exc}")
            _kill_process_tree(proc)

    state["running"] = False
    state["process"] = None
    state["current_full_report_path"] = ""
    state["current_target_path"] = ""
    return "ok"

@app.get("/events")
def get_events():
    state = _state()
    game_path = _latest_game_display_path()
    cached_game_path = _server_cached_html_path(game_path, _get_sid()) if game_path else str(state.get("cached_full_report_path") or "")

    target_path = _target_dfa_path()
    target_url = _path_to_url(target_path, "raw") if target_path else ""

    text = "\n".join(state["logs"])
    budget_exhausted = _is_tool_budget_exhausted(text)

    events = _events_from_logs()
    result = _run_result()

    report_url = url_for("analysis_artifact_route") if (cached_game_path or result in {"won", "lost", "crashed", "stopped"}) else ""

    return jsonify({
        "running": state["running"],
        "result": result,
        "failure_type": _failure_type_from_events(events) if result == "lost" else "",
        "events": events,
        "target_url": target_url,
        "budget_exhausted": budget_exhausted,
        "tool_request_started": _has_started_agent_tool_call(text),
        "full_report_url": report_url,
        "save_note": _result_save_note_from_logs(text),
        "token_metrics": _latest_token_metrics_from_logs(text),
    })

def _zoomed_html_document(content: str, scale: float = 0.28) -> str:
    """Wrap a DFA HTML artifact so it appears zoomed out with no scrollbars."""
    escaped = html_lib.escape(content, quote=True)
    virtual_w = 1600
    virtual_h = 1200
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset=\"utf-8\">
<style>
  html, body {{
    margin:0;
    padding:0;
    width:100%;
    height:100%;
    overflow:hidden !important;
    background:#ffffff;
  }}
  #viewport {{
    position:relative;
    width:100vw;
    height:100vh;
    overflow:hidden !important;
    background:#ffffff;
  }}
  #stage {{
    width:{virtual_w}px;
    height:{virtual_h}px;
    position:absolute;
    left:50%;
    top:50%;
    transform:translate(-50%, -50%) scale({scale});
    transform-origin:center center;
    overflow:hidden !important;
  }}
  #inner {{
    width:{virtual_w}px;
    height:{virtual_h}px;
    border:0;
    display:block;
    background:#ffffff;
    overflow:hidden !important;
  }}
</style>
<script>
function disableAllScrollbars() {{
  try {{
    document.documentElement.style.overflow = 'hidden';
    document.body.style.overflow = 'hidden';
    const frame = document.getElementById('inner');
    frame.setAttribute('scrolling', 'no');
    const doc = frame.contentDocument || frame.contentWindow.document;
    if(doc.documentElement) doc.documentElement.style.overflow = 'hidden';
    if(doc.body) doc.body.style.overflow = 'hidden';
  }} catch(e) {{}}
}}
function scheduleNoScroll() {{
  disableAllScrollbars();
  setTimeout(disableAllScrollbars, 50);
  setTimeout(disableAllScrollbars, 200);
  setTimeout(disableAllScrollbars, 600);
  setTimeout(disableAllScrollbars, 1200);
}}
window.addEventListener('load', () => {{
  const frame = document.getElementById('inner');
  frame.addEventListener('load', scheduleNoScroll);
  scheduleNoScroll();
}});
</script>
</head>
<body>
<div id=\"viewport\"><div id=\"stage\"><iframe id=\"inner\" scrolling=\"no\" srcdoc=\"{escaped}\"></iframe></div></div>
</body>
</html>"""

@app.get("/lib/<path:filename>")
def serve_pyvis_lib(filename):
    """Serve local PyVis/vis-network helper files referenced by generated DFA HTML.

    Some generated HTML files reference assets such as:
      /lib/bindings/utils.js

    Flask does not serve that folder automatically, so we expose the likely
    artifact lib folders explicitly. This removes the repeated 404 requests.
    """
    candidate_dirs = [
        ROOT / "runs" / "DFA" / "lib",
        ROOT / "runs" / "lib",
        ROOT / "lib",
    ]

    for directory in candidate_dirs:
        target = directory / filename
        if target.exists() and target.is_file():
            return send_from_directory(directory, filename)

    return Response("Library asset not found", status=404)


def _resolve_html_artifact_path(path: str, sid: str | None = None) -> Path | None:
    sid = sid or _get_sid()
    raw = unquote(html_lib.unescape(str(path or "").strip()))
    if raw.startswith("file:///"):
        raw = raw[8:]
    elif raw.startswith("file://"):
        raw = raw[7:]

    candidates: list[Path] = []
    if raw:
        try:
            candidates.append(Path(raw).resolve())
        except Exception:
            pass

    base = os.path.basename(raw.replace("\\", "/")) if raw else ""
    search_roots = [
        ROOT / "runs" / "server_html_cache" / sid,
        ROOT / _session_output_dir(sid),
    ]
    for root in search_roots:
        try:
            if base:
                candidates.extend(root.rglob(base))
            else:
                candidates.extend((root / "html").glob("session_*.html"))
        except Exception:
            pass

    for c in candidates:
        try:
            c = c.resolve()
            if c.exists() and c.is_file() and c.suffix.lower() == ".html":
                return c
        except Exception:
            continue
    return None


def _html_artifact_response_for_path(path: str, view: str) -> Response:
    p = _resolve_html_artifact_path(path)
    if not p:
        return Response("HTML artifact not found", status=404)

    content = p.read_text(encoding="utf-8", errors="replace")

    if view in {"candidate", "raw"}:
        content = _zoomed_html_document(content, scale=0.30)

    elif view == "target":
        matches = []
        for m in re.finditer(
            r"<iframe\b[^>]*\bsrcdoc=(['\"])(.*?)\1[^>]*>",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        ):
            matches.append(m.group(2))

        if matches:
            content = html_lib.unescape(matches[0])

        content = _zoomed_html_document(content, scale=0.30)

    elif view == "full":
        # Keep the report itself as the iframe document. Do not wrap it in an
        # additional iframe/srcdoc. Add a base only for relative non-embedded
        # links, while the session HTML remains fully visible.
        base_href = request.url_root.rstrip("/") + "/"
        base_tag = f'<base href="{html_lib.escape(base_href, quote=True)}">'
        if "<base " not in content.lower():
            if re.search(r"<head[^>]*>", content, flags=re.IGNORECASE):
                content = re.sub(r"(<head[^>]*>)", r"\1" + base_tag, content, count=1, flags=re.IGNORECASE)
            else:
                content = base_tag + content

    return Response(content, mimetype="text/html; charset=utf-8")


@app.get("/analysis_artifact")
def analysis_artifact_route():
    sid = _get_sid()
    state = _state(sid)
    game_path = _latest_game_display_path()
    cached_game_path = _server_cached_html_path(game_path, sid) if game_path else str(state.get("cached_full_report_path") or "")
    if not cached_game_path:
        return Response("<!doctype html><html><body style='font-family:Arial;padding:24px'>Analysis HTML is not available yet.</body></html>", mimetype="text/html; charset=utf-8")
    return _html_artifact_response_for_path(cached_game_path, "full")


@app.get("/html_artifact")
def html_artifact_route():
    path = unquote(request.args.get("path", ""))
    view = unquote(request.args.get("view", "full"))
    try:
        return _html_artifact_response_for_path(path, view)
    except Exception as exc:
        return Response(f"Failed to read artifact: {type(exc).__name__}: {exc}", status=500)


@app.get("/download_results_zip")
def download_results_zip():
    sid = _get_sid()
    try:
        zip_path = _make_results_zip(sid)
        return send_file(zip_path, as_attachment=True, download_name="automata_run_results.zip", mimetype="application/zip")
    except Exception as exc:
        return Response(f"Failed to create results ZIP: {type(exc).__name__}: {exc}", status=500)


@app.get("/logs")
def get_logs():
    return Response("\n".join(_state()["logs"]), mimetype="text/plain; charset=utf-8")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)
