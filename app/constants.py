"""
Responsible for:

- Defining global configuration constants used across the project
- Configuring models, strategies, tool prompts, game prompts, and learning hints
- Providing experiment parameters such as DFA size, alphabet size, and query-budget controls
- Defining non-informative scoring thresholds and runtime behavior constants
- Serving as the central configuration and shared constants module for the full system
"""
from L_star import LStarStrategy
from TTT import TTTStrategy
from utils import vocabulary_hint


# -----------------------------------------------------------------------------
# Required CLI parameters
# -----------------------------------------------------------------------------
# The following values must be provided through the command line:
#   --api-provider
#   --model-name
#   --api-key
#   --n-states
#   --seed
# They intentionally do not have defaults in constants.py.


# -----------------------------------------------------------------------------
# Optional CLI parameters with defaults
# -----------------------------------------------------------------------------
# These defaults are used unless the user overrides them from the command line.

OUTPUT_DIR = "runs"
EXPERIMENT_CSV = "results.csv"
COUNTEREXAMPLE_MODE = "deterministic short counterexample"
ALPHABET_SIZE = 2
ALGORITHM_APPROXIMATION_RATIO = 2

GAME_PROMPT = (
    "You are a language model tasked with reconstructing an unknown Deterministic Finite Automaton (DFA).\n"
    "A fixed target DFA exists, but its structure is hidden from you.\n"
    "\n"
    "You can learn about the target DFA only through:\n"
    "- External tools available to you.\n"
    "- Your own internal reasoning.\n"
    "\n"
    "Game rules:\n"
    "- You have a total budget of {MAX_CALLS} tool calls.\n"
    "- Each tool invocation counts as exactly one call.\n"
    "- If you exceed the allowed number of calls, your final score is 0.\n"
    "- On your final remaining call, you must invoke the evaluation tool.\n"
    "\n"
    "Scoring policy:\n"
    "- Using fewer tool calls while still reconstructing the correct DFA yields a higher score. Therefore, tools should be invoked efficiently and purposefully to obtain novel and informative results."
    "- If you believe you have already identified the correct DFA structure, you should immediately invoke the evaluation tool,\n"
    "  even if you still have remaining calls available.\n"
    "\n"
    "STRICT OUTPUT POLICY (MANDATORY):\n"
    "- You MUST NOT output any explanations, reasoning, comments, or natural language text.\n"
    "- You MUST output ONLY TOOL_ACTION blocks (Do NOT output any text) and exactly ONE TOOL_ACTION block.\n"
    "- Each TOOL_ACTION block must contain a single JSON object with exactly:\n"
    '  {"tool_name": "<tool_name>", "input": { ... }}\n'
    "- Never include call_count (it is added externally).\n"
    "\n"
    "Example:\n"
    "<TOOL_ACTION>\n"
    '{ "tool_name": "is_word_in_language", "input": { "word": "ab" } }\n'
    "</TOOL_ACTION>\n"
    "# IMPORTANT: The <TOOL_ACTION> block must include exactly one tool call. Each block represents a single tool invocation, so multiple tool calls—either to the same tool or to different tools—are not allowed within the same block.\n"
)


# -----------------------------------------------------------------------------
# Fixed constants that cannot be changed from the command line
# -----------------------------------------------------------------------------

# Controls whether the passive-learner analysis is also computed for the
# classical baseline strategies L* and TTT. This is intentionally a fixed
# code constant and is not exposed as a CLI argument.
#
# True  -> compute LStar_* / TTT_* passive learner columns and HTML blocks.
# False -> skip those computations; the CSV keeps the columns with -1/FALSE
#          defaults, and the HTML shows X for the skipped baseline values.
COMPUTE_BASELINE_PASSIVE_LEARNERS = True

# Controls language-similarity metric computation for baseline strategies.
# The LLM metric is always computed from its EQ hypotheses.
# True  -> compute LLM + L* + TTT similarity columns/HTML links.
# False -> compute only LLM similarity columns/HTML links.
COMPUTE_LANGUAGE_SIMILARITY_FOR_BASELINES = False

# Maximum word length k used by the symmetric-difference similarity metric:
# similarity = 1 - |L(hypothesis) Δ L(target)| / |Sigma^{<=k}|.
LANGUAGE_SIMILARITY_MAX_WORD_LENGTH = 100

# Number of background workers used for runtime language-similarity jobs.
# 1 means the game continues immediately, while similarity computations run
# one-at-a-time in the background.
LANGUAGE_SIMILARITY_BACKGROUND_WORKERS = 1

GEMINI_PROVIDER_EXTRA_CONFIG = {
    "require_context_cache": True,
    "include_thoughts": True,
    "first_cache_create_step": 10,
    "context_cache_ttl": "1800s",
}

PROVIDER_EXTRA_CONFIGS = {
    "gemini": GEMINI_PROVIDER_EXTRA_CONFIG,
}

STRATEGIES = [
    LStarStrategy,
    TTTStrategy,
]

TOOL_PROMPTS = {
    "is_word_in_language": (
        "Input JSON: { tool_name: 'is_word_in_language', call_count: int, input: { word: str } }\n"
        "Output JSON: { tool_name: 'is_word_in_language', call_count: int, error: str|null, output: { word: str, accepted: bool }|null }\n"
        "Checks whether the given word is accepted by the game's DFA. If the input word is the empty string \"\", the returned output.word will be \"ε\" (epsilon)."
    ),
    "evaluate_dfa_candidate": (
        "Input JSON: { tool_name: 'evaluate_dfa_candidate', call_count: int, input: { candidate_dfa: { "
        "states: [str], alphabet: [str], start_state: str, accept_states: [str], transitions: [[str, str, str]] } } }\n"
        "Output JSON: { tool_name: 'evaluate_dfa_candidate', call_count: int, error: str|null, output: { score: float, optimal: bool, witness_word: str }|null }\n"
        "Evaluates whether the provided candidate DFA is equivalent to the game's target DFA. "
        "If equivalent, output.optimal=true, output.score=1.0, and output.witness_word='' . "
        "If not equivalent, output.optimal=false, output.score=0.0, and output.witness_word is a counterexample word accepted by exactly one of the DFAs. "
        "If the witness word is the empty string, output.witness_word will be 'ε'. "
        "The counterexample also has a learning role: it shows where the candidate DFA fails and helps refine the next hypothesis."
    ),
}

HINT_VOCABULARY = "vocabulary"

HINTS = {
    HINT_VOCABULARY: HINT_VOCABULARY,
}

HINT_PROMPTS = {
    HINT_VOCABULARY: vocabulary_hint,
}

MAX_TOOL_LIMIT = 10000
NONINFORMATIVE_SCORE_STOP = 20000000000
NONINF_MQ_DUPLICATE_POINTS = 5
NONINF_MQ_HIT_EQ_WITNESS_POINTS = 4
NONINF_EQ_DUPLICATE_POINTS = 5
NONINF_EQ_CONTRADICTS_PREV_MQ_POINTS = 2
NONINF_EQ_CONTRADICTS_PREV_EQ_WITNESS_POINTS = 1
