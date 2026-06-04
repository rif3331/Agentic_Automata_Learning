"""
Responsible for:

- Serving as the main entrypoint for launching DFA-learning experiments
- Building random DFAs, configuring the interactive game, and setting query budgets
- Initializing tools, hints, runtime components, and LLM model configuration
- Managing experiment output paths for HTML and CSV exports
- Running the interactive chat session and coordinating full experiment execution
"""
from __future__ import annotations

from output_paths import set_output_dir
import sys
import argparse
from pathlib import Path
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfa_factory import make_random_dfa, make_regex_dfa
from game_format import GameFormat
from llm_runtime import LLM_interactive_game
from model_factory import build_llm_from_provider_model
from dfa_class import MinimalDFA
from constants import (
    PROVIDER_EXTRA_CONFIGS,
    OUTPUT_DIR,
    EXPERIMENT_CSV,
    COUNTEREXAMPLE_MODE,
    ALPHABET_SIZE,
    ALGORITHM_APPROXIMATION_RATIO,
    GAME_PROMPT,
)
from utils import (
    get_tools,
    get_effective_model_config,
    get_model_display_name,
    parse_model_name_and_inline_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="RUN DFA LEARNING EXPERIMENT"
    )

    parser.add_argument(
        "--n-states",
        type=int,
        default=2,
        help="Number of DFA states when --target-source=dataset."
    )

    parser.add_argument(
        "--alphabet-size",
        type=int,
        default=ALPHABET_SIZE,
        help="ALPHABET_SIZE"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed when --target-source=dataset."
    )

    parser.add_argument(
        "--api-provider",
        required=True,
        help="Required. API provider, e.g. gemini/openai/deepseek/together/openrouter/groq, or any OpenAI-compatible server URL."
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help="Required. Exact model name inside the selected provider, e.g. gemini-3.1-pro-preview or model-id(key=value)."
    )

    parser.add_argument(
        "--api-key",
        required=True,
        help="Required. Single API key for the selected API provider."
    )


    parser.add_argument(
        "--experiment-csv",
        default=EXPERIMENT_CSV,
        help="CSV FILE NAME"
    )

    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="DIRECTORY FOR ALL OUTPUT FILES"
    )

    parser.add_argument(
        "--counterexample-mode",
        default=COUNTEREXAMPLE_MODE,
        help="Counterexample mode."
    )

    parser.add_argument(
        "--algorithm-approximation-ratio",
        type=float,
        default=ALGORITHM_APPROXIMATION_RATIO,
        help="Multiplier applied to the minimum baseline query count to set the LLM tool-call budget."
    )

    parser.add_argument(
        "--game-prompt",
        default=GAME_PROMPT,
        help="Game prompt template. Defaults to constants.GAME_PROMPT."
    )

    parser.add_argument(
        "--target-source",
        choices=["dataset", "regex"],
        default="dataset",
        help="Choose the hidden target DFA source: dataset/random DFA or user regex."
    )

    parser.add_argument(
        "--regex",
        default="",
        help="Regular expression used when --target-source=regex. Example: '(a|b)*abb'."
    )

    # Required model identity/authentication arguments are intentionally CLI-only.
    # There are no provider-specific API-key arguments.

    return parser.parse_args()


def get_counterexample_mode(counterexample_mode: str) -> int:
    if counterexample_mode == "deterministic short counterexample":
        return 0

    if counterexample_mode == "minimal counterexample":
        return 1



    raise ValueError(f"Unknown counterexample_mode: {counterexample_mode}")


def get_minimal_counterexample(counterexample_mode: str) -> bool:
    mode = get_counterexample_mode(counterexample_mode)
    return mode == 1


def ensure_random_params(args):
    if args.n_states is None:
        while True:
            ans = input(
                "\nEnter number of states n_states (positive integer): "
            ).strip()

            try:
                args.n_states = int(ans)
                if args.n_states > 0:
                    break
            except Exception:
                pass

            print("Invalid n_states. Try again.")

    if args.alphabet_size is None:
        while True:
            ans = input(
                "Enter alphabet size alphabet_size (positive integer): "
            ).strip()

            try:
                args.alphabet_size = int(ans)
                if args.alphabet_size > 0:
                    break
            except Exception:
                pass

            print("Invalid alphabet_size. Try again.")

    if args.seed is None:
        while True:
            ans = input("Enter seed (integer): ").strip()

            try:
                args.seed = int(ans)
                break
            except Exception:
                pass

            print("Invalid seed. Try again.")

    return args.n_states, args.alphabet_size, args.seed


def get_min_total_queries(strategy_results):
    return min(
        result.total_queries
        for run in strategy_results
        for result in run.values()
    )


def round_half_up_to_int(value) -> int:
    """Round budget like the analysis code, and always return int.

    This matters because the runtime enforces the tool budget only when
    max_tool_calls is an int. If we pass 12.0, the prompt says 12.0 but
    the guard does not stop the model.
    """
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def build_run_paths(args):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    html_dir = output_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    html_path = html_dir / f"session_{ts}.html"
    csv_path = output_dir / args.experiment_csv

    conversation_link = f"file:///{html_path.as_posix()}"

    return html_path, csv_path, conversation_link


def build_game(args):
    target_source = str(getattr(args, "target_source", "dataset") or "dataset").strip().lower()

    if target_source == "regex":
        regex_value = str(getattr(args, "regex", "") or "").strip()
        if not regex_value:
            raise ValueError("--regex is required when --target-source=regex")
        raw_dfa = make_regex_dfa(regex_value)
        chosen = f"regex({regex_value})"
    else:
        n_states_val, alphabet_size_val, seed_val = ensure_random_params(args)
        raw_dfa = make_random_dfa(
            n_states=n_states_val,
            alphabet_size=alphabet_size_val,
            seed=seed_val,
        )
        chosen = (
            f"random(n_states={n_states_val},"
            f"alphabet_size={alphabet_size_val},"
            f"seed={seed_val})"
        )

    counterexample_mode_value = get_counterexample_mode(args.counterexample_mode)

    mdfa = MinimalDFA.from_dfa(
        raw_dfa,
        run_strategy=True,
        minimal_counterexample=get_minimal_counterexample(
            args.counterexample_mode
        ),
        counterexample_max_extra_len=3,
        counterexample_mode=counterexample_mode_value,
    )

    target_draw = mdfa.draw()
    print(f"TARGET_DFA_LINK::{target_draw}", flush=True)

    tool_classes = get_tools()

    hint_defaults = {
        "vocabulary": True,
    }

    def ask_hint(hint_key: str, display: str) -> bool:
        return hint_defaults.get(hint_key, False)

    def ask(tool_cls) -> bool:
        name = getattr(tool_cls, "__name__", "")
        return name in (
            "IsWordInLanguageTool",
            "EvaluateDFACandidateTool",
        )

    min_baseline_queries = get_min_total_queries(mdfa.strategy_results)
    optimal_query_count = round_half_up_to_int(
        min_baseline_queries * args.algorithm_approximation_ratio
    )

    print(f"TOOL_BUDGET::{optimal_query_count}", flush=True)
    print(f"BASELINE_MIN_TOOL_CALLS::{min_baseline_queries}", flush=True)
    print(f"BUDGET_FACTOR::{args.algorithm_approximation_ratio}", flush=True)

    game = GameFormat.interactive_build(
        dfa=mdfa,
        tool_classes=tool_classes,
        max_tool_calls=optimal_query_count,
        ask=ask,
        ask_hint=ask_hint,
        game_prompt=args.game_prompt,
    )

    return game, chosen


def main():
    args = parse_args()
    set_output_dir(args.output_dir)

    game, chosen = build_game(args)
    bridge = LLM_interactive_game(game=game)

    provider = str(args.api_provider or "").strip().lower()

    # Build the final model configuration from constants.py.
    # Provider-level defaults, such as the Gemini cache/thought settings, are
    # applied first. Exact-model additions are then applied on top.
    base_model_name, _inline_model_config = parse_model_name_and_inline_config(args.model_name)
    gen_config = get_effective_model_config(provider, args.model_name, PROVIDER_EXTRA_CONFIGS)
    gen_config["display_model_name"] = get_model_display_name(provider, args.model_name)

    llm = build_llm_from_provider_model(
        api_provider=args.api_provider,
        model_name=base_model_name,
        api_key=args.api_key,
        mode="chat",
        system_prompt=None,
        generation_config=gen_config,
        safety_settings=None,
    )

    html_path, csv_path, conversation_link = build_run_paths(args)

    bridge.run_with_chat_session(
        llm,
        verbose=True,
        export_csv_path=str(csv_path),
        export_html_path=str(html_path),
        conversation_link=conversation_link,
        seed=str(chosen),
    )


if __name__ == "__main__":
    main()