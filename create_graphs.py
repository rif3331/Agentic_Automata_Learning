from pathlib import Path
import argparse
import ast
import json
import math
import re
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d


# =========================
# USER CONFIGURATION
# =========================

CSV_PATH = "results.csv"

MODEL_COLUMN = "llm_model"
STATE_COLUMN = "number_of_states"
LLM_TOTAL_QUERIES_COLUMN = "llm_total_queries"
MAX_TOOL_CALLS_COLUMN = "max_tool_calls"
STRATEGIES_COLUMN = "strategies"

# If your table has llm_symdiff_similarity_by_step, the code will use it.
# Otherwise, it will fall back to llm_symdiff_distance_by_step and convert distance to similarity.
SIMILARITY_BY_STEP_COLUMN = "llm_symdiff_similarity_by_step"
DISTANCE_BY_STEP_COLUMN = "llm_symdiff_distance_by_step"

X_AXIS_MODE = 1
PAIR_X_GROUPS = True
X_GROUP_PAIR_START = 2
X_GROUP_SIZE = 2

# 1 = success, 2 = best similarity.
# The script below automatically creates:
#   graph 1: success graph, equivalent to Y_AXIS_MODE = 1
#   graph 2: average tool calls graph
#   graph 3: similarity graph, equivalent to Y_AXIS_MODE = 2
Y_AXIS_MODE = 1

SKIP_X_ROWS = False
ONLY_SUCCESSFUL_RUNS = False
MAX_SEED = 20000000
FILTER_LLM_QUERIES_MODE = 1
# 1 = for failed LLM runs marked X, use max_tool_calls in the tool-call graph.
# 2 = ignore failed LLM runs in the tool-call graph.

SHOW_ERROR_BARS = True
SHOW_SUCCESS_ERROR_BARS = False
SHOW_TOOL_ERROR_BARS = True
SHOW_BASELINE_ERROR_BARS = False
SHOW_LEGENDS = True

FIG_WIDTH = 13
FIG_HEIGHT = 6.5
BAR_GROUP_WIDTH = 0.82
SUCCESS_Y_MAX = 105
SIMILARITY_Y_MIN = 20
SIMILARITY_Y_MAX = 105
SIMILARITY_RANDOM_BASELINE = 50
SIMILARITY_RANDOM_BASELINE_LABEL = "random language proposals"
TOOL_Y_MAX = None

SAVE_PDF = True
PDF_OUTPUT_PATH = None
PDF_OUTPUT_FILENAME = "graphs.pdf"

# Extra graph: overall success rate under classical-budget multipliers.
# This is the graph controlled by USE_SUCCESS_FACTOR_SWEEP_GRAPH in the original code.
SHOW_SUCCESS_FACTOR_SWEEP_GRAPH = True
SUCCESS_FACTOR_VALUES = [1.0, 1.25, 1.5, 1.75, 2.0]
SUCCESS_FACTOR_SWEEP_TITLE = "Overall success rate by classical-budget multiplier"
SUCCESS_FACTOR_SWEEP_X_LABEL = "Query budget factor"
SUCCESS_FACTOR_SWEEP_Y_LABEL = "Success rate (%)"

# Extra graph from the analysis code: all runs context-window / token growth.
# This graph is saved as an additional page in the same PDF.
SHOW_CONTEXT_WINDOW_GRAPH = True
CONTEXT_GRAPH_TITLE = "Context Window Analysis"
CONTEXT_GRAPH_X_LABEL = "Call number"
CONTEXT_GRAPH_Y_LABEL = "Total tokens per call"
STEPS_COLUMN = "step_metrics"
TOKEN_COLUMN = "game_token_tuple"
TOTAL_TIME_COLUMN = "total_game_time_s"

# Extra analysis summary graphs from the analysis script.
# 6 = total tokens by number of states, 5 = total runtime by number of states.
SHOW_ALL_ROWS_TOKENS_BY_STATES_GRAPH = True
SHOW_ALL_ROWS_RUNTIME_BY_STATES_GRAPH = True
STATE_X_JITTER = 0.0

SHOW_NONINFORMATIVE_BY_STEP_GRAPH = True
NONINFORMATIVE_STEP_MAX_STEP = 100
NONINFORMATIVE_STEP_SMOOTHING_SIGMA = 2.0
NONINFORMATIVE_STEP_GRAPH_TITLE = "Non-informative query rate by interaction step"
NONINFORMATIVE_STEP_X_LABEL = "Step number"
NONINFORMATIVE_STEP_Y_LABEL = "← Non-informative queries (%)"

DIRECTLY_NONINFO_KEYS = [
    "MQ DUPLICATE STEPS",
    "EQ DUPLICATE STEPS",
    "EQ CONTRADICTS PREVIOUS MQ",
]

LSTAR_LABEL = "L*"
TTT_LABEL = "TTT"

SUCCESS_GRAPH_TITLE = "Model performance as the number of states in the minimal DFA increases"
SUCCESS_Y_LABEL = "Success rate (%) →"
SIMILARITY_GRAPH_TITLE = "Best hypothesis similarity as the number of states in the minimal DFA increases"
SIMILARITY_Y_LABEL = "Best hypothesis similarity (%) →"
TOOL_GRAPH_TITLE = "Average tool calls as the number of states in the minimal DFA increases"
TOOL_Y_LABEL = "Avg Tool calls (← lower is better)"
X_AXIS_LABEL = "Number of States in the Minimal DFA"

# Colors are assigned by order of first appearance in the CSV.
# If two model labels differ only by text inside parentheses, they use the same base color,
# but later variants are lightened so the relationship is visible.
BASE_MODEL_COLOR_PALETTE = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "gold",
    "cyan",
]
VARIANT_LIGHTEN_STEPS = [0.0, 0.30, 0.50, 0.65, 0.78]

# Optional mapping from raw model names to paper names.
# If your CSV already contains the paper names, this mapping is not required.
MODEL_DISPLAY_NAMES = {
    "gemini:gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
    "gemini:gemini-3-flash-preview": "Gemini-3-Flash-Preview (thinking)",
    "gemini:gemini-3.1-flash-lite-preview": "Gemini-3.1-Flash-Lite-Preview",
    "openai:gpt-5.4": "GPT-5.4 (without thinking)",
    "openai:gpt-5.4-thinking": "GPT-5.4 (thinking)",
    "deepseek:deepseek-v4-pro": "Deepseek-v4-Pro",
    "deepseek:deepseek-reasoner": "DeepSeek-V3.2 (thinking)",
    "together:menagedreef_265f/meta-llama/Llama-3.3-70B-Instruct-Turbo-5f2c0da6": "Llama-3.3-70B-Instruct-Turbo",
}

EXCLUDE_MODELS = []


HOVER_CONNECTIONS = []


def add_hover_cursor(artists, artist_to_text):
    artists = [artist for artist in artists if artist is not None]
    if not artists:
        return

    fig = artists[0].figure
    ax = artists[0].axes
    annotation = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(14, 14),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", ec="0.35", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="0.35"),
        fontsize=9,
        zorder=20,
    )
    annotation.set_visible(False)

    def resolve_text(artist, event):
        text_getter = artist_to_text.get(artist)
        if callable(text_getter):
            try:
                return text_getter(event)
            except TypeError:
                return text_getter()
        return text_getter

    def on_move(event):
        if event.inaxes != ax:
            if annotation.get_visible():
                annotation.set_visible(False)
                fig.canvas.draw_idle()
            return

        for artist in artists:
            try:
                contains, info = artist.contains(event)
            except Exception:
                contains, info = False, {}
            if not contains:
                continue

            text = resolve_text(artist, event)
            if not text:
                continue
            annotation.xy = (event.xdata, event.ydata)
            annotation.set_text(text)
            annotation.set_visible(True)
            fig.canvas.draw_idle()
            return

        if annotation.get_visible():
            annotation.set_visible(False)
            fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("motion_notify_event", on_move)
    HOVER_CONNECTIONS.append((fig, cid, annotation))


def fmt_num(value, digits=2):
    try:
        value = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(value):
        return "nan"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def set_percent_axis_ticks(ax, y_min, y_max):
    ax.set_ylim(y_min, y_max)
    ticks = [t for t in range(0, 101, 10) if y_min <= t <= 100]
    ax.set_yticks(ticks)


# =========================
# HELPERS
# =========================

def short_model_name(name):
    name = str(name).strip()
    return MODEL_DISPLAY_NAMES.get(name, name)


def is_missing_model(value):
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def strip_parentheses(label):
    base = re.sub(r"\s*\([^)]*\)", "", str(label)).strip()
    base = re.sub(r"\s+", " ", base)
    return base


def lighten_color(color, amount):
    rgb = np.array(to_rgb(color), dtype=float)
    white = np.array([1.0, 1.0, 1.0], dtype=float)
    return tuple(rgb + (white - rgb) * amount)


def assign_model_colors(model_labels_in_order):
    base_to_color = OrderedDict()
    base_variant_seen = defaultdict(list)
    label_to_color = {}

    for label in model_labels_in_order:
        base = strip_parentheses(label)
        if base not in base_to_color:
            base_to_color[base] = BASE_MODEL_COLOR_PALETTE[len(base_to_color) % len(BASE_MODEL_COLOR_PALETTE)]

        if label not in base_variant_seen[base]:
            base_variant_seen[base].append(label)

        variant_index = base_variant_seen[base].index(label)
        lighten_amount = VARIANT_LIGHTEN_STEPS[min(variant_index, len(VARIANT_LIGHTEN_STEPS) - 1)]
        label_to_color[label] = lighten_color(base_to_color[base], lighten_amount)

    return label_to_color


def normalize_x_value(v):
    if pd.isna(v):
        return np.nan
    f = float(v)
    if abs(f - round(f)) < 1e-9:
        return int(round(f))
    return f


def pair_x_group_value(v):
    if pd.isna(v):
        return np.nan
    n = int(float(v))
    if not PAIR_X_GROUPS:
        return n
    if n < X_GROUP_PAIR_START:
        return n
    return X_GROUP_PAIR_START + X_GROUP_SIZE * ((n - X_GROUP_PAIR_START) // X_GROUP_SIZE)


def format_x_tick_label(x):
    f = float(x)
    if abs(f - round(f)) < 1e-9:
        n = int(round(f))
        if PAIR_X_GROUPS and n >= X_GROUP_PAIR_START:
            return f"{n}-{n + X_GROUP_SIZE - 1}"
        return str(n)
    return f"{f:g}"


def parse_strategy_queries(strategy_value, strategy_name):
    if pd.isna(strategy_value):
        return np.nan
    m = re.search(rf"{re.escape(strategy_name)}=(\d+)", str(strategy_value))
    return float(m.group(1)) if m else np.nan


def round_half_up(value):
    try:
        value = float(value)
    except Exception:
        return np.nan
    if not np.isfinite(value):
        return np.nan
    return int(math.floor(value + 0.5))


def min_lstar_ttt_strategy_queries(strategy_value):
    lstar_queries = parse_strategy_queries(strategy_value, "LStarStrategy")
    ttt_queries = parse_strategy_queries(strategy_value, "TTTStrategy")
    values = [v for v in [lstar_queries, ttt_queries] if np.isfinite(v)]
    if not values:
        return np.nan
    return min(values)


def success_by_strategy_multiplier_with_factor(row, factor):
    """
    Success under a classical-strategy-based budget.
    budget = round_half_up(min(L*, TTT) * factor).
    Rows where llm_total_queries is X are counted as failure.
    """
    llm_total_raw = str(row.get(LLM_TOTAL_QUERIES_COLUMN, "")).strip()
    if llm_total_raw == "" or llm_total_raw.lower() == "nan":
        return np.nan
    if llm_total_raw.upper() == "X":
        return 0

    llm_total_queries = pd.to_numeric(llm_total_raw, errors="coerce")
    if pd.isna(llm_total_queries) or not np.isfinite(llm_total_queries):
        return np.nan

    min_strategy_queries = min_lstar_ttt_strategy_queries(row.get(STRATEGIES_COLUMN, np.nan))
    if not np.isfinite(min_strategy_queries):
        return np.nan

    multiplied_budget = round_half_up(min_strategy_queries * factor)
    if not np.isfinite(multiplied_budget):
        return np.nan

    return 1 if multiplied_budget >= float(llm_total_queries) else 0


def extract_seed(seed_str):
    if pd.isna(seed_str):
        return np.nan
    m = re.search(r"seed=(\d+)", str(seed_str))
    return int(m.group(1)) if m else np.nan


def is_success(value):
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    return 0 if s.upper() == "X" else 1


def parse_llm_total_queries(value, max_tool_calls):
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    if s.upper() == "X":
        if FILTER_LLM_QUERIES_MODE == 2:
            return np.nan
        return pd.to_numeric(max_tool_calls, errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def parse_list_like(value):
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(s)
            if isinstance(parsed, (list, tuple)):
                return parsed
        except Exception:
            pass
    return []


def load_obj(value):
    if pd.isna(value):
        return {}
    if isinstance(value, dict):
        return value
    s = str(value).strip()
    if not s:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
    return {}


def get_float(d, key, default=0.0):
    try:
        value = d.get(key, default)
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def extract_total_tokens_per_call(row):
    """
    Matches the logic from the analysis code:
    total per call = new input tokens + cache input tokens + output tokens.
    """
    steps_obj = load_obj(row.get(STEPS_COLUMN))
    step_items = []
    for key, value in steps_obj.items():
        try:
            step_num = int(key)
        except Exception:
            continue
        if not isinstance(value, dict):
            continue

        input_tokens = get_float(value, "input_tokens")
        thoughts_tokens = get_float(value, "thoughts_tokens")
        output_visible_tokens = get_float(value, "output_visible_tokens")
        cache_history_tokens = get_float(value, "cache_history_tokens")

        input_without_cache = max(input_tokens - cache_history_tokens, 0.0)
        output_total = thoughts_tokens + output_visible_tokens
        total_tokens = input_without_cache + cache_history_tokens + output_total
        step_items.append((step_num, total_tokens))

    step_items.sort(key=lambda item: item[0])
    x = np.array([item[0] for item in step_items], dtype=int)
    y = np.array([item[1] for item in step_items], dtype=float)
    return x, y


def best_similarity_from_row(row):
    if SIMILARITY_BY_STEP_COLUMN in row.index:
        values = []
        for item in parse_list_like(row.get(SIMILARITY_BY_STEP_COLUMN)):
            try:
                v = float(item)
            except Exception:
                continue
            if np.isfinite(v) and v != -1:
                values.append(v)
        if values:
            return max(values)

    if DISTANCE_BY_STEP_COLUMN in row.index:
        values = []
        for item in parse_list_like(row.get(DISTANCE_BY_STEP_COLUMN)):
            try:
                v = float(item)
            except Exception:
                continue
            if np.isfinite(v) and v != -1:
                values.append(v)
        if values:
            return 1.0 - min(values)

    return np.nan


def build_x_group(df):
    df = df.copy()
    df[STATE_COLUMN] = pd.to_numeric(df[STATE_COLUMN], errors="coerce")
    df = df.dropna(subset=[STATE_COLUMN]).copy()
    df[STATE_COLUMN] = df[STATE_COLUMN].astype(int)
    df["x_group"] = df[STATE_COLUMN].apply(normalize_x_value).apply(pair_x_group_value)
    df = df.dropna(subset=["x_group"]).copy()
    return df


def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = [STATE_COLUMN, STRATEGIES_COLUMN]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if SKIP_X_ROWS and LLM_TOTAL_QUERIES_COLUMN in df.columns:
        df = df[df[LLM_TOTAL_QUERIES_COLUMN].astype(str).str.strip().str.upper() == "X"].copy()

    if ONLY_SUCCESSFUL_RUNS and LLM_TOTAL_QUERIES_COLUMN in df.columns:
        df = df[df[LLM_TOTAL_QUERIES_COLUMN].astype(str).str.strip().str.upper() != "X"].copy()

    if "seed" in df.columns:
        df["seed_value"] = df["seed"].apply(extract_seed)
        df = df[(df["seed_value"].isna()) | (df["seed_value"] <= MAX_SEED)].copy()

    df = build_x_group(df)
    return df


def model_rows(df):
    required_cols = [MODEL_COLUMN, LLM_TOTAL_QUERIES_COLUMN]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for model graphs: {missing}")

    out = df.copy()
    out = out[~out[MODEL_COLUMN].apply(is_missing_model)].copy()
    out["model_label"] = out[MODEL_COLUMN].apply(short_model_name)
    out = out[~out["model_label"].isin(EXCLUDE_MODELS)].copy()
    return out


def get_model_order_by_first_appearance(df_models):
    order = []
    seen = set()
    for label in df_models["model_label"].tolist():
        if label not in seen:
            seen.add(label)
            order.append(label)
    return order


def get_model_order_by_binary_success(agg_success, x_order, fallback_order):
    """
    Sort models from best to worst by binary success.
    Primary key is the last difficulty group. If there is a tie, compare the
    previous group, and so on. If still tied, keep CSV first-appearance order.
    """
    fallback_index = {model: i for i, model in enumerate(fallback_order)}

    def score_tuple(model):
        model_df = agg_success[agg_success["model_label"] == model]
        values = []
        for group in reversed(x_order):
            row = model_df[model_df["x_group"] == group]
            if row.empty:
                values.append(-1.0)
            else:
                values.append(float(row.iloc[0]["mean_value"]))
        values.append(-fallback_index.get(model, 10**9))
        return tuple(values)

    return sorted(fallback_order, key=score_tuple, reverse=True)


def get_x_order(df):
    # Keep every state value that appears in the data, including state=1.
    # Earlier versions hid state=1 in grouped mode, but current reports should
    # show it whenever corresponding rows exist.
    return sorted(df["x_group"].dropna().unique(), key=float)


def aggregate_success(df_models):
    df_success = df_models.copy()
    df_success["success"] = df_success[LLM_TOTAL_QUERIES_COLUMN].apply(is_success)
    df_success = df_success.dropna(subset=["success"]).copy()
    return (
        df_success.groupby(["model_label", "x_group"], as_index=False)
        .agg(mean_value=("success", "mean"), std_value=("success", "std"), runs=("success", "size"))
    )


def aggregate_similarity(df_models):
    df_sim = df_models.copy()
    df_sim["similarity"] = df_sim.apply(best_similarity_from_row, axis=1)
    df_sim = df_sim.dropna(subset=["similarity"]).copy()
    return (
        df_sim.groupby(["model_label", "x_group"], as_index=False)
        .agg(mean_value=("similarity", "mean"), std_value=("similarity", "std"), runs=("similarity", "size"))
    )


def aggregate_tool_calls(df_models):
    if MAX_TOOL_CALLS_COLUMN not in df_models.columns:
        raise ValueError(f"Missing required column for tool graph: {MAX_TOOL_CALLS_COLUMN}")

    df_tool = df_models.copy()
    df_tool[MAX_TOOL_CALLS_COLUMN] = pd.to_numeric(df_tool[MAX_TOOL_CALLS_COLUMN], errors="coerce")
    df_tool["tool_calls"] = df_tool.apply(
        lambda row: parse_llm_total_queries(row[LLM_TOTAL_QUERIES_COLUMN], row[MAX_TOOL_CALLS_COLUMN]),
        axis=1,
    )
    df_tool = df_tool.dropna(subset=["tool_calls"]).copy()
    return (
        df_tool.groupby(["model_label", "x_group"], as_index=False)
        .agg(mean_value=("tool_calls", "mean"), std_value=("tool_calls", "std"), runs=("tool_calls", "size"))
    )


def aggregate_baselines_from_all_rows(df_all):
    # Important: L* and TTT are computed from every row in the table, not from a specific model.
    df_base = df_all.copy()
    df_base["lstar_queries"] = df_base[STRATEGIES_COLUMN].apply(lambda v: parse_strategy_queries(v, "LStarStrategy"))
    df_base["ttt_queries"] = df_base[STRATEGIES_COLUMN].apply(lambda v: parse_strategy_queries(v, "TTTStrategy"))

    lstar = (
        df_base.dropna(subset=["lstar_queries"])
        .groupby("x_group", as_index=False)
        .agg(mean=("lstar_queries", "mean"), std=("lstar_queries", "std"), runs=("lstar_queries", "size"))
    )
    ttt = (
        df_base.dropna(subset=["ttt_queries"])
        .groupby("x_group", as_index=False)
        .agg(mean=("ttt_queries", "mean"), std=("ttt_queries", "std"), runs=("ttt_queries", "size"))
    )

    lstar["std"] = lstar["std"].fillna(0.0)
    ttt["std"] = ttt["std"].fillna(0.0)
    return lstar, ttt


def draw_grouped_bar_graph(agg, model_order, model_colors, x_order, title, y_label, y_max, value_multiplier=100.0):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    x = np.arange(len(x_order), dtype=float)
    n_models = max(len(model_order), 1)
    fig.canvas.draw()
    axis_px = max(float(ax.bbox.width), 1.0)
    data_span = max(float(len(x_order)), 1.0)
    max_bar_width_data = (30.0 * data_span / axis_px) / 0.92
    bar_width = min(0.11, BAR_GROUP_WIDTH / n_models, max_bar_width_data)
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_width

    hover_artists = []
    artist_to_text = {}

    for i, model in enumerate(model_order):
        model_df = agg[agg["model_label"] == model]
        y_values = []
        std_values = []
        runs_values = []

        for group in x_order:
            row = model_df[model_df["x_group"] == group]
            if row.empty:
                y_values.append(np.nan)
                std_values.append(np.nan)
                runs_values.append(np.nan)
            else:
                y_values.append(float(row.iloc[0]["mean_value"]) * value_multiplier)
                std_values.append(float(row.iloc[0]["std_value"] if pd.notna(row.iloc[0]["std_value"]) else 0.0) * value_multiplier)
                runs_values.append(float(row.iloc[0]["runs"]))

        xpos = x + offsets[i]
        bars = ax.bar(
            xpos,
            y_values,
            width=bar_width * 0.92,
            label=model,
            color=model_colors[model],
            edgecolor="black",
            linewidth=0.55,
            alpha=0.9,
            zorder=3,
        )

        for group, bar, yi, si, runs in zip(x_order, bars, y_values, std_values, runs_values):
            if not np.isfinite(yi):
                continue
            hover_artists.append(bar)
            artist_to_text[bar] = (
                f"{model}\n"
                f"states={format_x_tick_label(group)}\n"
                f"value={fmt_num(yi)}%\n"
                f"std={fmt_num(si)}%\n"
                f"runs={fmt_num(runs, 0)}"
            )

        if SHOW_ERROR_BARS and SHOW_SUCCESS_ERROR_BARS:
            for xi, yi, si in zip(xpos, y_values, std_values):
                if not np.isfinite(yi) or not np.isfinite(si):
                    continue
                lower = min(si, yi)
                upper = min(si, max(0.0, y_max - yi)) if y_max is not None else si
                ax.errorbar(
                    xi,
                    yi,
                    yerr=np.array([[lower], [upper]]),
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=3,
                    capthick=1.0,
                    zorder=4,
                )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(X_AXIS_LABEL, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([format_x_tick_label(g) for g in x_order])
    if len(x_order) == 1:
        # Cap the visual width of a single vertical bar at about 30 pixels.
        # This keeps one-run / one-state graphs from filling the whole plot.
        fig.canvas.draw()
        axis_px = max(float(ax.bbox.width), 1.0)
        drawn_bar_data_width = max(float(bar_width * 0.92), 1e-6)
        full_group_data_width = max(float(bar_width * (n_models - 1) + drawn_bar_data_width), drawn_bar_data_width)
        span_for_30px = drawn_bar_data_width * axis_px / 30.0
        span = max(span_for_30px, full_group_data_width * 1.35, 1.0)
        ax.set_xlim(-span / 2.0, span / 2.0)
    elif len(x_order) > 1:
        ax.set_xlim(-0.5, len(x_order) - 0.5)
    if y_max is not None:
        set_percent_axis_ticks(ax, 0, y_max)
    ax.grid(axis="y", alpha=0.28)
    ax.set_axisbelow(True)

    if SHOW_LEGENDS:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=8, frameon=True)

    add_hover_cursor(hover_artists, artist_to_text)
    plt.tight_layout()
    return fig, ax

def draw_tool_calls_graph(agg_tool, model_order, model_colors, x_order, agg_lstar, agg_ttt):
    y_max = TOOL_Y_MAX
    if y_max is None:
        vals = []
        vals.extend((agg_tool["mean_value"] + agg_tool["std_value"].fillna(0.0)).dropna().tolist())
        vals.extend((agg_lstar["mean"] + agg_lstar["std"].fillna(0.0)).dropna().tolist())
        vals.extend((agg_ttt["mean"] + agg_ttt["std"].fillna(0.0)).dropna().tolist())
        y_max = max(vals) * 1.12 if vals else 100

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    x = np.arange(len(x_order), dtype=float)
    n_models = max(len(model_order), 1)
    bar_width = min(0.11, BAR_GROUP_WIDTH / n_models)
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_width

    hover_artists = []
    artist_to_text = {}

    for i, model in enumerate(model_order):
        model_df = agg_tool[agg_tool["model_label"] == model]
        y_values = []
        std_values = []
        runs_values = []

        for group in x_order:
            row = model_df[model_df["x_group"] == group]
            if row.empty:
                y_values.append(np.nan)
                std_values.append(np.nan)
                runs_values.append(np.nan)
            else:
                y_values.append(float(row.iloc[0]["mean_value"]))
                std_values.append(float(row.iloc[0]["std_value"] if pd.notna(row.iloc[0]["std_value"]) else 0.0))
                runs_values.append(float(row.iloc[0]["runs"]))

        xpos = x + offsets[i]
        bars = ax.bar(
            xpos,
            y_values,
            width=bar_width * 0.92,
            label=model,
            color=model_colors[model],
            edgecolor="black",
            linewidth=0.55,
            alpha=0.9,
            zorder=3,
        )

        for group, bar, yi, si, runs in zip(x_order, bars, y_values, std_values, runs_values):
            if not np.isfinite(yi):
                continue
            hover_artists.append(bar)
            artist_to_text[bar] = (
                f"{model}\n"
                f"states={format_x_tick_label(group)}\n"
                f"avg tool calls={fmt_num(yi)}\n"
                f"std={fmt_num(si)}\n"
                f"runs={fmt_num(runs, 0)}"
            )

        # The tool-call graph shows variance/std bars when enabled.
        if SHOW_ERROR_BARS and SHOW_TOOL_ERROR_BARS:
            for xi, yi, si in zip(xpos, y_values, std_values):
                if not np.isfinite(yi) or not np.isfinite(si):
                    continue
                lower = min(si, yi)
                upper = min(si, max(0.0, y_max - yi))
                ax.errorbar(
                    xi,
                    yi,
                    yerr=np.array([[lower], [upper]]),
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=3,
                    capthick=1.0,
                    zorder=4,
                )

    x_centers = {group: idx for idx, group in enumerate(x_order)}

    def draw_baseline(agg_base, label, color):
        for _, row in agg_base.iterrows():
            group = row["x_group"]
            if group not in x_centers:
                continue
            center = x_centers[group]
            y = float(row["mean"])
            std = float(row["std"] if pd.notna(row["std"]) else 0.0)
            runs = float(row["runs"] if pd.notna(row["runs"]) else 0.0)
            line = ax.plot(
                [center - BAR_GROUP_WIDTH / 2, center + BAR_GROUP_WIDTH / 2],
                [y, y],
                color=color,
                linestyle=(0, (3, 3)),
                linewidth=1.7,
                zorder=5,
            )[0]
            hover_artists.append(line)
            artist_to_text[line] = (
                f"{label}\n"
                f"states={format_x_tick_label(group)}\n"
                f"avg tool calls={fmt_num(y)}\n"
                f"std={fmt_num(std)}\n"
                f"runs={fmt_num(runs, 0)}"
            )

            if SHOW_BASELINE_ERROR_BARS:
                ax.errorbar(
                    center - BAR_GROUP_WIDTH / 2 + 0.06,
                    y,
                    yerr=np.array([[min(std, y)], [min(std, max(0.0, y_max - y))]]),
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=3,
                    capthick=1.0,
                    zorder=6,
                )

    draw_baseline(agg_ttt, TTT_LABEL, "black")
    draw_baseline(agg_lstar, LSTAR_LABEL, "0.45")

    ax.set_title(TOOL_GRAPH_TITLE, fontsize=14, fontweight="bold")
    ax.set_xlabel(X_AXIS_LABEL, fontsize=12)
    ax.set_ylabel(TOOL_Y_LABEL, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([format_x_tick_label(g) for g in x_order])
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", alpha=0.28)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    handles.extend([
        Line2D([0], [0], color="black", linestyle=(0, (3, 3)), linewidth=1.7, label=TTT_LABEL),
        Line2D([0], [0], color="0.45", linestyle=(0, (3, 3)), linewidth=1.7, label=LSTAR_LABEL),
    ])
    labels.extend([TTT_LABEL, LSTAR_LABEL])

    if SHOW_LEGENDS:
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=8, frameon=True)

    add_hover_cursor(hover_artists, artist_to_text)
    plt.tight_layout()
    return fig, ax


def draw_similarity_line_graph(agg_similarity, model_order, model_colors, x_order):
    """Draw similarity like the appendix figure: lines + shaded one-std region."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    x = np.arange(len(x_order), dtype=float)

    hover_artists = []
    artist_to_text = {}

    for model in model_order:
        model_df = agg_similarity[agg_similarity["model_label"] == model]
        y_values = []
        std_values = []
        runs_values = []
        group_values = []

        for group in x_order:
            row = model_df[model_df["x_group"] == group]
            if row.empty:
                y_values.append(np.nan)
                std_values.append(np.nan)
                runs_values.append(np.nan)
            else:
                y_values.append(float(row.iloc[0]["mean_value"]) * 100.0)
                std_values.append(float(row.iloc[0]["std_value"] if pd.notna(row.iloc[0]["std_value"]) else 0.0) * 100.0)
                runs_values.append(float(row.iloc[0]["runs"]))
            group_values.append(group)

        y = np.array(y_values, dtype=float)
        std = np.array(std_values, dtype=float)
        runs_arr = np.array(runs_values, dtype=float)
        valid = np.isfinite(y)
        if not valid.any():
            continue

        xv = x[valid]
        yv = y[valid]
        sv = np.where(np.isfinite(std[valid]), std[valid], 0.0)
        rv = runs_arr[valid]
        gv = np.array(group_values, dtype=object)[valid]
        color = model_colors[model]

        ax.fill_between(
            xv,
            np.maximum(SIMILARITY_Y_MIN, yv - sv),
            np.minimum(SIMILARITY_Y_MAX, yv + sv),
            color=color,
            alpha=0.15,
            linewidth=0,
            zorder=2,
        )
        line = ax.plot(
            xv,
            yv,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=model,
            zorder=3,
        )[0]
        hover_artists.append(line)
        def make_similarity_hover(model_name, xv_local, yv_local, sv_local, rv_local, gv_local):
            def _text(event):
                if event.xdata is None or len(xv_local) == 0:
                    idx = 0
                else:
                    idx = int(np.argmin(np.abs(xv_local - event.xdata)))
                return (
                    f"{model_name}\n"
                    f"states={format_x_tick_label(gv_local[idx])}\n"
                    f"best similarity={fmt_num(yv_local[idx])}%\n"
                    f"std={fmt_num(sv_local[idx])}%\n"
                    f"runs={fmt_num(rv_local[idx], 0)}"
                )
            return _text

        artist_to_text[line] = make_similarity_hover(model, xv, yv, sv, rv, gv)

    baseline = ax.axhline(
        SIMILARITY_RANDOM_BASELINE,
        color="gray",
        linestyle=(0, (3, 3)),
        linewidth=1.5,
        zorder=1,
    )
    hover_artists.append(baseline)
    artist_to_text[baseline] = f"{SIMILARITY_RANDOM_BASELINE_LABEL}\nsimilarity={fmt_num(SIMILARITY_RANDOM_BASELINE)}%"

    ax.text(
        x.mean() if len(x) else 0,
        SIMILARITY_RANDOM_BASELINE - 2.0,
        SIMILARITY_RANDOM_BASELINE_LABEL,
        ha="center",
        va="top",
        fontsize=9,
        color="gray",
    )

    ax.set_title(SIMILARITY_GRAPH_TITLE, fontsize=14, fontweight="bold")
    ax.set_xlabel(X_AXIS_LABEL, fontsize=12)
    ax.set_ylabel(SIMILARITY_Y_LABEL, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([format_x_tick_label(g) for g in x_order])
    if len(x_order) == 1:
        # Keep a single-state similarity graph visually narrow instead of
        # stretching one point across the whole plotting area.
        fig.canvas.draw()
        axis_px = max(float(ax.bbox.width), 1.0)
        span_for_30px = axis_px / 30.0
        span = max(span_for_30px, 1.0)
        ax.set_xlim(-span / 2.0, span / 2.0)
    elif len(x_order) > 1:
        ax.set_xlim(-0.5, len(x_order) - 0.5)
    set_percent_axis_ticks(ax, SIMILARITY_Y_MIN, SIMILARITY_Y_MAX)
    ax.grid(True, alpha=0.28)
    ax.set_axisbelow(True)

    if SHOW_LEGENDS:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=8, frameon=True)

    add_hover_cursor(hover_artists, artist_to_text)
    plt.tight_layout()
    return fig, ax


def draw_context_window_graph(df_models, model_colors):
    """Draw the context-window analysis graph from the supplied analysis code."""
    if STEPS_COLUMN not in df_models.columns:
        return None, None

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    seen_labels = set()
    y_max_values = []
    hover_artists = []
    artist_to_text = {}

    for _, row in df_models.iterrows():
        x, y = extract_total_tokens_per_call(row)
        if len(x) == 0:
            continue

        model = str(row.get("model_label", row.get(MODEL_COLUMN, "unknown")))
        color = model_colors.get(model, None)
        label = model if model not in seen_labels else "_nolegend_"
        seen_labels.add(model)

        line = ax.plot(
            x,
            y,
            linewidth=1.0,
            color=color,
            label=label,
            alpha=0.85,
        )[0]
        hover_artists.append(line)

        number_of_states = row.get(STATE_COLUMN, "")
        total_queries = row.get(LLM_TOTAL_QUERIES_COLUMN, "")
        max_tool_calls = row.get(MAX_TOOL_CALLS_COLUMN, "")
        if len(y) > 0:
            y_max_values.append(np.nanmax(y))

        def make_context_hover(model_name, x_local, y_local, states_value, total_q, max_calls):
            def _text(event):
                if event.xdata is None or len(x_local) == 0:
                    idx = 0
                else:
                    idx = int(np.argmin(np.abs(x_local - event.xdata)))
                return (
                    f"{model_name}\n"
                    f"call={fmt_num(x_local[idx], 0)}\n"
                    f"tokens={fmt_num(y_local[idx], 0)}\n"
                    f"states={states_value}\n"
                    f"llm_total_queries={total_q}\n"
                    f"max_tool_calls={max_calls}"
                )
            return _text

        artist_to_text[line] = make_context_hover(model, x, y, number_of_states, total_queries, max_tool_calls)

    if not hover_artists:
        plt.close(fig)
        return None, None

    ax.set_title(CONTEXT_GRAPH_TITLE, fontsize=14, fontweight="bold")
    ax.set_xlabel(CONTEXT_GRAPH_X_LABEL, fontsize=12)
    ax.set_ylabel(CONTEXT_GRAPH_Y_LABEL, fontsize=12)
    if y_max_values:
        ax.set_ylim(0, max(max(y_max_values) * 1.10, 10))
    ax.grid(True, alpha=0.28)
    ax.set_axisbelow(True)

    if SHOW_LEGENDS and seen_labels:
        ax.legend(title="Models", loc="upper left", fontsize=8, frameon=True)

    add_hover_cursor(hover_artists, artist_to_text)
    plt.tight_layout()
    return fig, ax


def compute_total_time_for_row(row):
    """Return total runtime for one row in seconds."""
    direct = pd.to_numeric(row.get(TOTAL_TIME_COLUMN, np.nan), errors="coerce")
    if pd.notna(direct) and np.isfinite(direct) and float(direct) > 0:
        return float(direct)

    steps_obj = load_obj(row.get(STEPS_COLUMN))
    total = 0.0
    for value in steps_obj.values():
        if isinstance(value, dict):
            total += get_float(value, "step_time_s", 0.0)
    return float(total)


def compute_total_tokens_for_row(row):
    """Return total token count for one row, matching SHOW_ALL_ROWS_SUM_GRAPH = 6."""
    token_obj = load_obj(row.get(TOKEN_COLUMN))
    total = get_float(token_obj, "total", 0.0)
    if total > 0:
        return float(total)

    # Fallback: sum the per-call totals if game_token_tuple.total is unavailable.
    _, y = extract_total_tokens_per_call(row)
    if len(y) == 0:
        return 0.0
    return float(np.sum(y))


def draw_all_rows_tokens_by_states_graph(df_models, model_colors):
    """Matches SHOW_ALL_ROWS_SUM_GRAPH = 6: Total tokens by number of states."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    seen_models = set()
    x_positions = []
    y_values_all = []
    hover_artists = []
    artist_to_text = {}

    for _, row in df_models.iterrows():
        model = str(row.get("model_label", row.get(MODEL_COLUMN, "unknown")))
        x_pos = pd.to_numeric(row.get(STATE_COLUMN, np.nan), errors="coerce")
        if pd.isna(x_pos) or not np.isfinite(x_pos):
            continue
        x_pos = float(x_pos)
        if STATE_X_JITTER:
            x_pos = x_pos + np.random.uniform(-STATE_X_JITTER, STATE_X_JITTER)

        total_tokens = compute_total_tokens_for_row(row)
        if not np.isfinite(total_tokens) or total_tokens <= 0:
            continue
        total_tokens_millions = total_tokens / 1_000_000.0

        label = model if model not in seen_models else "_nolegend_"
        seen_models.add(model)
        scatter = ax.scatter(
            [x_pos],
            [total_tokens_millions],
            s=60,
            color=model_colors.get(model, None),
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
            label=label,
        )
        hover_artists.append(scatter)
        artist_to_text[scatter] = (
            f"{model}\n"
            f"states={fmt_num(row.get(STATE_COLUMN, np.nan), 0)}\n"
            f"total tokens={fmt_num(total_tokens, 0)}\n"
            f"total tokens={fmt_num(total_tokens_millions)}M\n"
            f"llm_total_queries={row.get(LLM_TOTAL_QUERIES_COLUMN, '')}\n"
            f"max_tool_calls={row.get(MAX_TOOL_CALLS_COLUMN, '')}"
        )

        x_positions.append(x_pos)
        y_values_all.append(total_tokens_millions)

    if not x_positions:
        plt.close(fig)
        return None, None

    unique_states = sorted({int(round(x)) for x in x_positions})
    ax.set_title("All Rows Total Tokens by Number of States", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of states", fontsize=12)
    ax.set_ylabel("Total tokens (millions)", fontsize=12)
    ax.set_xticks(unique_states)
    ax.set_xlim(min(unique_states) - 0.5, max(unique_states) + 0.5)
    ax.set_ylim(0, max(max(y_values_all) * 1.1, 0.01))
    ax.grid(True, alpha=0.28)
    ax.set_axisbelow(True)
    if SHOW_LEGENDS and seen_models:
        ax.legend(title="Models", loc="upper left", fontsize=8, frameon=True)
    add_hover_cursor(hover_artists, artist_to_text)
    plt.tight_layout()
    return fig, ax


def draw_all_rows_runtime_by_states_graph(df_models, model_colors):
    """Matches SHOW_ALL_ROWS_SUM_GRAPH = 5: Total runtime by number of states."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    seen_models = set()
    x_positions = []
    y_values_all = []
    hover_artists = []
    artist_to_text = {}

    for _, row in df_models.iterrows():
        model = str(row.get("model_label", row.get(MODEL_COLUMN, "unknown")))
        x_pos = pd.to_numeric(row.get(STATE_COLUMN, np.nan), errors="coerce")
        if pd.isna(x_pos) or not np.isfinite(x_pos):
            continue
        x_pos = float(x_pos)
        if STATE_X_JITTER:
            x_pos = x_pos + np.random.uniform(-STATE_X_JITTER, STATE_X_JITTER)

        total_seconds = compute_total_time_for_row(row)
        if not np.isfinite(total_seconds) or total_seconds <= 0:
            continue
        total_hours = total_seconds / 3600.0

        label = model if model not in seen_models else "_nolegend_"
        seen_models.add(model)
        scatter = ax.scatter(
            [x_pos],
            [total_hours],
            s=60,
            color=model_colors.get(model, None),
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
            label=label,
        )
        hover_artists.append(scatter)
        artist_to_text[scatter] = (
            f"{model}\n"
            f"states={fmt_num(row.get(STATE_COLUMN, np.nan), 0)}\n"
            f"runtime={fmt_num(total_hours)} hours\n"
            f"runtime={fmt_num(total_seconds, 0)} seconds\n"
            f"llm_total_queries={row.get(LLM_TOTAL_QUERIES_COLUMN, '')}\n"
            f"max_tool_calls={row.get(MAX_TOOL_CALLS_COLUMN, '')}"
        )

        x_positions.append(x_pos)
        y_values_all.append(total_hours)

    if not x_positions:
        plt.close(fig)
        return None, None

    unique_states = sorted({int(round(x)) for x in x_positions})
    ax.set_title("All Rows Total Runtime by Number of States", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of states", fontsize=12)
    ax.set_ylabel("Total runtime (hours)", fontsize=12)
    ax.set_xticks(unique_states)
    ax.set_xlim(min(unique_states) - 0.5, max(unique_states) + 0.5)
    ax.set_ylim(0, max(max(y_values_all) * 1.1, 0.1))
    ax.grid(True, alpha=0.28)
    ax.set_axisbelow(True)
    if SHOW_LEGENDS and seen_models:
        ax.legend(title="Models", loc="upper left", fontsize=8, frameon=True)
    add_hover_cursor(hover_artists, artist_to_text)
    plt.tight_layout()
    return fig, ax


def aggregate_success_factor_sweep(df_models):
    sweep_rows = []

    for factor in SUCCESS_FACTOR_VALUES:
        df_factor = df_models.copy()
        df_factor["factor_success"] = df_factor.apply(
            lambda row: success_by_strategy_multiplier_with_factor(row, factor),
            axis=1,
        )
        df_factor = df_factor.dropna(subset=["factor_success", "model_label"]).copy()
        df_factor["factor_success"] = pd.to_numeric(df_factor["factor_success"], errors="coerce")
        df_factor = df_factor.dropna(subset=["factor_success"]).copy()
        if df_factor.empty:
            continue

        grouped = (
            df_factor.groupby("model_label", as_index=False)
            .agg(
                mean_value=("factor_success", "mean"),
                std_value=("factor_success", "std"),
                successes=("factor_success", "sum"),
                runs=("factor_success", "size"),
            )
        )
        grouped["std_value"] = grouped["std_value"].fillna(0.0)
        grouped["factor"] = factor
        sweep_rows.append(grouped)

    if not sweep_rows:
        return pd.DataFrame(columns=["model_label", "mean_value", "std_value", "successes", "runs", "factor"])
    return pd.concat(sweep_rows, ignore_index=True)


def draw_success_factor_sweep_graph(agg_sweep, model_order, model_colors):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    factors = list(SUCCESS_FACTOR_VALUES)
    x = np.arange(len(factors), dtype=float)
    n_models = max(len(model_order), 1)
    bar_width = min(0.11, BAR_GROUP_WIDTH / n_models)
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_width

    hover_artists = []
    artist_to_text = {}

    for i, model in enumerate(model_order):
        model_df = agg_sweep[agg_sweep["model_label"] == model]
        y_values = []
        std_values = []
        success_values = []
        runs_values = []

        for factor in factors:
            row = model_df[np.isclose(model_df["factor"].astype(float), float(factor))]
            if row.empty:
                y_values.append(np.nan)
                std_values.append(np.nan)
                success_values.append(np.nan)
                runs_values.append(np.nan)
            else:
                r = row.iloc[0]
                y_values.append(float(r["mean_value"]) * 100.0)
                std_values.append(float(r["std_value"] if pd.notna(r["std_value"]) else 0.0) * 100.0)
                success_values.append(float(r["successes"]))
                runs_values.append(float(r["runs"]))

        xpos = x + offsets[i]
        bars = ax.bar(
            xpos,
            y_values,
            width=bar_width * 0.92,
            label=model,
            color=model_colors[model],
            edgecolor="black",
            linewidth=0.55,
            alpha=0.9,
            zorder=3,
        )

        for factor, bar, yi, si, successes, runs in zip(factors, bars, y_values, std_values, success_values, runs_values):
            if not np.isfinite(yi):
                continue
            hover_artists.append(bar)
            artist_to_text[bar] = (
                f"{model}\n"
                f"factor={factor:g}\n"
                f"success={fmt_num(successes, 0)}/{fmt_num(runs, 0)}\n"
                f"rate={fmt_num(yi)}%\n"
                f"std={fmt_num(si)}%"
            )

    ax.set_title(SUCCESS_FACTOR_SWEEP_TITLE, fontsize=14, fontweight="bold")
    ax.set_xlabel(SUCCESS_FACTOR_SWEEP_X_LABEL, fontsize=12)
    ax.set_ylabel(SUCCESS_FACTOR_SWEEP_Y_LABEL, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{factor:g}" for factor in factors])
    set_percent_axis_ticks(ax, 0, SUCCESS_Y_MAX)
    ax.grid(axis="y", alpha=0.28)
    ax.set_axisbelow(True)

    if SHOW_LEGENDS:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=8, frameon=True)

    add_hover_cursor(hover_artists, artist_to_text)
    plt.tight_layout()
    return fig, ax



def parse_noninformative_info(value):
    return load_obj(value)


def extract_current_step(item):
    if isinstance(item, (list, tuple)) and len(item) >= 1:
        try:
            return int(item[0])
        except Exception:
            return None
    try:
        return int(item)
    except Exception:
        return None


def collect_directly_noninfo_steps(noninfo_dict):
    steps = set()
    if not isinstance(noninfo_dict, dict):
        return steps

    for key in DIRECTLY_NONINFO_KEYS:
        values = noninfo_dict.get(key, [])
        if not isinstance(values, list):
            continue
        for item in values:
            step = extract_current_step(item)
            if step is not None:
                steps.add(step)

    return steps


def build_noninformative_query_points(df_models):
    if "noninformative_queries_info" not in df_models.columns or STEPS_COLUMN not in df_models.columns:
        return pd.DataFrame()

    rows = []

    for row_index, row in df_models.iterrows():
        model_label = str(row.get("model_label", row.get(MODEL_COLUMN, "")))
        noninfo_dict = parse_noninformative_info(row.get("noninformative_queries_info", ""))
        step_metrics = load_obj(row.get(STEPS_COLUMN, ""))

        if not isinstance(step_metrics, dict):
            continue

        directly_noninfo_steps = collect_directly_noninfo_steps(noninfo_dict)

        for step_key, metrics in step_metrics.items():
            if not isinstance(metrics, dict):
                continue
            try:
                step_metrics_key_int = int(step_key)
            except Exception:
                continue

            # In the original graph, query_step = step_metrics key + 1.
            query_step = step_metrics_key_int + 1

            rows.append({
                "row_index": row_index,
                "model_label": model_label,
                "query_step": int(query_step),
                "is_directly_noninformative": bool(query_step in directly_noninfo_steps),
            })

    return pd.DataFrame(rows)


def aggregate_noninformative_by_step(df_models):
    points = build_noninformative_query_points(df_models)
    if points.empty:
        return pd.DataFrame()

    grouped = (
        points
        .groupby(["model_label", "query_step"])["is_directly_noninformative"]
        .agg(
            directly_noninformative_queries="sum",
            total_queries="count",
        )
        .reset_index()
    )

    grouped["directly_noninformative_rate"] = (
        grouped["directly_noninformative_queries"] / grouped["total_queries"]
    )
    grouped["directly_noninformative_percent"] = (
        grouped["directly_noninformative_rate"] * 100.0
    )

    if NONINFORMATIVE_STEP_MAX_STEP is not None:
        grouped = grouped[grouped["query_step"] <= NONINFORMATIVE_STEP_MAX_STEP].copy()

    return grouped


def smooth_noninformative_values(y_values):
    y_values = np.asarray(y_values, dtype=float)
    if NONINFORMATIVE_STEP_SMOOTHING_SIGMA is None or NONINFORMATIVE_STEP_SMOOTHING_SIGMA <= 0:
        return y_values
    if len(y_values) < 3:
        return y_values
    return gaussian_filter1d(
        y_values,
        sigma=NONINFORMATIVE_STEP_SMOOTHING_SIGMA,
        mode="nearest",
    )


def draw_noninformative_by_step_graph(df_models, model_order, model_colors):
    step_df = aggregate_noninformative_by_step(df_models)
    if step_df.empty:
        return None, None

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    hover_artists = []
    artist_to_text = {}

    for model in model_order:
        model_df = step_df[step_df["model_label"] == model].copy()
        if model_df.empty:
            continue

        model_df = model_df.sort_values("query_step")
        x_values = model_df["query_step"].to_numpy()
        raw_y_values = model_df["directly_noninformative_percent"].to_numpy()
        y_values = smooth_noninformative_values(raw_y_values)
        total_queries = model_df["total_queries"].to_numpy()
        noninfo_queries = model_df["directly_noninformative_queries"].to_numpy()

        # No markers: line only.
        line = ax.plot(
            x_values,
            y_values,
            linewidth=2.0,
            color=model_colors.get(model, None),
            label=model,
        )[0]

        hover_artists.append(line)

        def make_noninfo_hover(model_name, x_local, y_local, raw_y_local, total_local, noninfo_local):
            def _text(event):
                if event.xdata is None or len(x_local) == 0:
                    idx = 0
                else:
                    idx = int(np.argmin(np.abs(x_local - event.xdata)))
                return (
                    f"{model_name}\n"
                    f"step={fmt_num(x_local[idx], 0)}\n"
                    f"non-informative={fmt_num(raw_y_local[idx])}%\n"
                    f"smoothed={fmt_num(y_local[idx])}%\n"
                    f"queries={fmt_num(noninfo_local[idx], 0)}/{fmt_num(total_local[idx], 0)}"
                )
            return _text

        artist_to_text[line] = make_noninfo_hover(
            model,
            x_values,
            y_values,
            raw_y_values,
            total_queries,
            noninfo_queries,
        )

    if not hover_artists:
        plt.close(fig)
        return None, None

    ax.set_title(NONINFORMATIVE_STEP_GRAPH_TITLE, fontsize=14, fontweight="bold")
    ax.set_xlabel(NONINFORMATIVE_STEP_X_LABEL, fontsize=12)
    ax.set_ylabel(NONINFORMATIVE_STEP_Y_LABEL, fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_yticks([t for t in range(0, 101, 10)])

    if NONINFORMATIVE_STEP_MAX_STEP is not None:
        ax.set_xlim(1, NONINFORMATIVE_STEP_MAX_STEP)

    ax.grid(True, alpha=0.28)
    ax.set_axisbelow(True)

    if SHOW_LEGENDS:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=2,
            fontsize=8,
            frameon=True,
        )

    add_hover_cursor(hover_artists, artist_to_text)
    plt.tight_layout()
    return fig, ax


def resolve_output_pdf_path(csv_path, output_pdf_path=PDF_OUTPUT_PATH):
    """
    If output_pdf_path is not provided, save the PDF next to the CSV file.
    Example:
        csv_path = runs/results.csv
        output  = runs/paper_three_graphs.pdf
    """
    if output_pdf_path:
        return Path(output_pdf_path)
    return Path(csv_path).expanduser().resolve().parent / PDF_OUTPUT_FILENAME





def order_task_outcome_models_by_success(tmp):
    summary_success = (
        tmp.groupby("model_label")["is_failure"]
        .agg(total="size", failures="sum")
        .reset_index()
    )
    summary_success["success_rate"] = (
        100.0 * (summary_success["total"] - summary_success["failures"]) / summary_success["total"]
    )

    first_order = {model: i for i, model in enumerate(tmp["model_label"].drop_duplicates().tolist())}

    return (
        summary_success
        .assign(first_order=lambda d: d["model_label"].map(first_order))
        .sort_values(
            by=["success_rate", "total", "first_order"],
            ascending=[False, False, True],
        )
        ["model_label"]
        .tolist()
    )


def draw_task_outcomes_stacked_bar(df_models):
    SUCCESS_LABEL = "Success Rate"
    REASONING_FAILURE_LABEL = "Reasoning Failure"
    PLANNING_FAILURE_LABEL = "Planning Failure"
    CATEGORY_ORDER = [SUCCESS_LABEL, REASONING_FAILURE_LABEL, PLANNING_FAILURE_LABEL]
    CATEGORY_COLORS = {
        SUCCESS_LABEL: "#6fbf73",
        REASONING_FAILURE_LABEL: "#f2b84b",
        PLANNING_FAILURE_LABEL: "#e57373",
    }

    tmp = df_models.copy()

    # Match the old correct script behavior: only chat rows participate if game_mode exists.
    if "game_mode" in tmp.columns:
        tmp = tmp[tmp["game_mode"].astype(str).str.lower() == "chat"].copy()

    tmp = tmp[~tmp[MODEL_COLUMN].apply(is_missing_model)].copy()

    tmp["is_failure"] = (
        tmp[LLM_TOTAL_QUERIES_COLUMN]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("X")
    )

    def is_minus_one(v):
        try:
            if pd.isna(v):
                return False
        except Exception:
            pass
        text = str(v).strip()
        if text.upper() == "X":
            return False
        try:
            return float(text) == -1.0
        except Exception:
            return text == "-1"

    # New results.csv uses FirstStep columns.
    # Old experiment tables use RPNI / EDSM / Blue-Fringe.
    passive_column_sets = [
        ("RPNI_FirstStep", "EDSM_FirstStep", "BlueFringe_FirstStep"),
        ("RPNI", "EDSM", "Blue-Fringe"),
    ]

    passive_cols = None
    for candidate_cols in passive_column_sets:
        if all(col in tmp.columns for col in candidate_cols):
            passive_cols = candidate_cols
            break

    if passive_cols is None:
        raise ValueError(
            "Missing passive learner columns. Expected either "
            "RPNI_FirstStep, EDSM_FirstStep, BlueFringe_FirstStep "
            "or RPNI, EDSM, Blue-Fringe."
        )

    rpni_col, edsm_col, blue_col = passive_cols

    # Planning failure:
    #   the LLM failed AND all passive learners failed on the accumulated evidence.
    # Reasoning failure:
    #   the LLM failed BUT at least one passive learner could infer the DFA.
    tmp["passive_algorithms_failed"] = (
        tmp[rpni_col].apply(is_minus_one)
        & tmp[edsm_col].apply(is_minus_one)
        & tmp[blue_col].apply(is_minus_one)
    )

    tmp["category"] = np.select(
        [
            ~tmp["is_failure"],
            tmp["is_failure"] & tmp["passive_algorithms_failed"],
            tmp["is_failure"] & ~tmp["passive_algorithms_failed"],
        ],
        [
            SUCCESS_LABEL,
            PLANNING_FAILURE_LABEL,
            REASONING_FAILURE_LABEL,
        ],
        default=PLANNING_FAILURE_LABEL,
    )

    summary = (
        tmp.groupby(["model_label", "category"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    # Keep the same model order used elsewhere in the PDF.
    models = order_task_outcome_models_by_success(tmp)

    # Keep the original graph width. Limit only the visual thickness of each
    # horizontal bar so a single model/run does not create an oversized bar.
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, max(5, len(models) * 0.6)))
    outcome_bar_height = 0.45

    left = np.zeros(len(models))

    for category in CATEGORY_ORDER:
        vals = []

        for model in models:
            row = summary[
                (summary["model_label"] == model)
                & (summary["category"] == category)
            ]

            count = int(row["count"].iloc[0]) if not row.empty else 0
            total = int(summary[summary["model_label"] == model]["count"].sum())
            vals.append(100.0 * count / total if total else 0.0)

        ax.barh(
            models,
            vals,
            left=left,
            height=outcome_bar_height,
            color=CATEGORY_COLORS[category],
            edgecolor="white",
            label=category,
        )
        left += np.array(vals)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of task instances")
    # No title requested for this graph.
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    if len(models) == 1:
        # Cap the visual thickness of a single horizontal bar at about 30 px.
        fig.canvas.draw()
        axis_px = max(float(ax.bbox.height), 1.0)
        span_for_30px = outcome_bar_height * axis_px / 30.0
        span = max(span_for_30px, 1.0)
        ax.set_ylim(span / 2.0, -span / 2.0)
    ax.set_xticks([t for t in range(0, 101, 10)])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3)

    plt.tight_layout()
    return fig, ax


def main(csv_path, output_pdf_path=PDF_OUTPUT_PATH):
    output_pdf_path = resolve_output_pdf_path(csv_path, output_pdf_path)
    df_all = load_and_prepare(csv_path)
    df_models = model_rows(df_all)

    if df_models.empty:
        raise ValueError("No model rows found. Rows with llm_model None / empty are skipped.")

    first_appearance_order = get_model_order_by_first_appearance(df_models)
    x_order = get_x_order(df_all)

    agg_success = aggregate_success(df_models)
    agg_tool = aggregate_tool_calls(df_models)
    agg_similarity = aggregate_similarity(df_models)
    agg_sweep = aggregate_success_factor_sweep(df_models)
    agg_lstar, agg_ttt = aggregate_baselines_from_all_rows(df_all)

    # Bar/legend order: best to worst by binary success, comparing from the
    # hardest group backward: 8-9, then 6-7, then 4-5, then 2-3.
    model_order = get_model_order_by_binary_success(agg_success, x_order, first_appearance_order)

    # Colors are still assigned by first appearance, so the first model seen in
    # the CSV is always blue, regardless of sorting in the graph.
    model_colors = assign_model_colors(first_appearance_order)

    fig_success, _ = draw_grouped_bar_graph(
        agg_success,
        model_order,
        model_colors,
        x_order,
        SUCCESS_GRAPH_TITLE,
        SUCCESS_Y_LABEL,
        SUCCESS_Y_MAX,
        value_multiplier=100.0,
    )

    fig_tool, _ = draw_tool_calls_graph(
        agg_tool,
        model_order,
        model_colors,
        x_order,
        agg_lstar,
        agg_ttt,
    )

    fig_similarity, _ = draw_grouped_bar_graph(
        agg_similarity,
        model_order,
        model_colors,
        x_order,
        SIMILARITY_GRAPH_TITLE,
        SIMILARITY_Y_LABEL,
        SIMILARITY_Y_MAX,
        value_multiplier=100.0,
    )

    extra_figures = []

    fig_task_outcomes, _ = draw_task_outcomes_stacked_bar(df_models)
    extra_figures.append(fig_task_outcomes)
    if SHOW_SUCCESS_FACTOR_SWEEP_GRAPH and not agg_sweep.empty:
        fig_sweep, _ = draw_success_factor_sweep_graph(
            agg_sweep,
            model_order,
            model_colors,
        )
        extra_figures.append(fig_sweep)

    if SHOW_CONTEXT_WINDOW_GRAPH:
        fig_context, _ = draw_context_window_graph(df_models, model_colors)
        if fig_context is not None:
            extra_figures.append(fig_context)

    if SHOW_ALL_ROWS_TOKENS_BY_STATES_GRAPH:
        fig_tokens_by_states, _ = draw_all_rows_tokens_by_states_graph(df_models, model_colors)
        if fig_tokens_by_states is not None:
            extra_figures.append(fig_tokens_by_states)

    if SHOW_ALL_ROWS_RUNTIME_BY_STATES_GRAPH:
        fig_runtime_by_states, _ = draw_all_rows_runtime_by_states_graph(df_models, model_colors)
        if fig_runtime_by_states is not None:
            extra_figures.append(fig_runtime_by_states)

    if SHOW_NONINFORMATIVE_BY_STEP_GRAPH:
        fig_noninfo_by_step, _ = draw_noninformative_by_step_graph(df_models, model_order, model_colors)
        if fig_noninfo_by_step is not None:
            extra_figures.append(fig_noninfo_by_step)

    all_figures = [fig_success, fig_tool, fig_similarity] + extra_figures

    if SAVE_PDF and output_pdf_path:
        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(output_pdf_path) as pdf:
            for fig in all_figures:
                pdf.savefig(fig, bbox_inches="tight")
        print(f"Saved PDF to: {output_pdf_path}")

    # Do not display graphs on screen; save them only to the PDF.
    for fig in all_figures:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the 3 main paper graphs from a CSV results table.")
    parser.add_argument("csv_path", nargs="?", default=CSV_PATH, help="Path to the CSV file")
    parser.add_argument(
        "--output-pdf",
        default=PDF_OUTPUT_PATH,
        help="Path for the combined output PDF. If omitted, saves next to the CSV file.",
    )
    args = parser.parse_args()
    main(args.csv_path, args.output_pdf)
