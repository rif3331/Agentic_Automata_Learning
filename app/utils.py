"""
Responsible for:

- Parsing model names and normalizing tool identifiers
- Tokenizing words based on DFA alphabets and vocabularies
- Converting words into sequence/DFA input representations
- Formatting and normalizing word display values, including epsilon handling
- Escaping and loading HTML helper content
"""
from typing import Optional
from typing import Any, Optional, Sequence
from pathlib import Path
from typing import (
    Any,Iterable,List,Optional,Protocol,Tuple,NamedTuple,runtime_checkable,Type,TypedDict,Dict,TYPE_CHECKING,Callable,Set
)

def parse_model_name(model_name: str) -> Tuple[str, str]:
    s = (model_name or "").strip()
    if ":" in s:
        p, m = s.split(":", 1)
        return p.strip().lower(), m.strip()
    return "gemini", s

def _tokenize_by_vocab(
    word: str,
    alphabet_set: set[str] | None,
) -> tuple[list[str] | None, str | None]:
    s = "" if word is None else str(word)
    s = s.strip()

    if s == "" or s == "ε":
        return ([], None)

    if any(ch.isspace() for ch in s):
        toks = [t for t in s.split() if t != ""]
        return (toks, None)

    if not alphabet_set:
        return (list(s), None)

    all_len_1 = True

    for sym in alphabet_set:
        if not isinstance(sym, str):
            all_len_1 = False
            break

        if len(sym) != 1:
            all_len_1 = False
            break

    if all_len_1:
        return (list(s), None)

    symbols = [
        x for x in alphabet_set
        if isinstance(x, str) and x != ""
    ]

    symbols.sort(
        key=lambda x: len(x),
        reverse=True
    )

    out: list[str] = []

    i = 0
    n = len(s)

    while i < n:
        matched = None

        for sym in symbols:
            if s.startswith(sym, i):
                matched = sym
                break

        if matched is None:
            return (None, s[i:i+1])

        out.append(matched)

        i += len(matched)

    return (out, None)


def _normalize_word_display(w: object) -> str:
    if w is None:
        return ""

    if isinstance(w, str):
        return "ε" if w == "" else w

    if isinstance(w, (list, tuple)):
        if len(w) == 0:
            return "ε"

        return " ".join(
            str(x)
            for x in w
        )

    return str(w)


def _escape_attr_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _load_html(html_or_path: str) -> str:
    p = Path(html_or_path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return html_or_path

def _word_to_seq(
    word: Any,
    alphabet: Optional[Sequence[Any]] = None,
) -> list[Any]:
    if word is None:
        return []

    if isinstance(word, str):
        toks, _bad = _tokenize_by_vocab(word, alphabet)
        if toks is None:
            return [word]
        return toks

    if isinstance(word, tuple):
        return list(word)

    if isinstance(word, list):
        return word

    try:
        return list(word)
    except Exception:
        return [word]


def _word_repr(seq: Sequence[Any]) -> str:
    if not seq:
        return ""
    return " ".join(str(x) for x in seq)


def normalize_tool_name(name: Any) -> str:
    s = "" if name is None else str(name)
    s = s.strip().replace("-", "_")
    s = "_".join(s.split())
    return s


def tokenize_voc(word: Any, alphabet: Optional[Iterable[str]]) -> List[str]:
    if word is None:
        return []
    s = str(word).strip()
    if s == "" or s == "ε":
        return []
    if any(ch.isspace() for ch in s):
        return [t for t in s.split() if t]
    if alphabet is None:
        return list(s)
    alpha = [a for a in alphabet if isinstance(a, str) and a]
    if not alpha:
        return list(s)
    if all(len(a) == 1 for a in alpha):
        return list(s)
    alpha_sorted = sorted(alpha, key=len, reverse=True)
    out: List[str] = []
    i = 0
    while i < len(s):
        m = None
        for sym in alpha_sorted:
            if s.startswith(sym, i):
                m = sym
                break
        if m is None:
            return [s]
        out.append(m)
        i += len(m)
    return out


def canonical_word(word: Any, alphabet: Optional[Iterable[str]]) -> str:
    toks = tokenize_voc(word, alphabet)
    return "ε" if not toks else " ".join(toks)


def word_to_dfa_input(word: Any, alphabet: Optional[Iterable[str]] = None) -> List[str]:
    return tokenize_voc(word, alphabet)

def normalize_config_provider(api_provider: str | None) -> str:
    provider = (api_provider or "").strip().lower()
    aliases = {
        "google": "gemini",
        "google_genai": "gemini",
        "google-ai": "gemini",
        "google_ai": "gemini",
    }
    return aliases.get(provider, provider)


def copy_config(config: dict | None) -> dict:
    """Return a deep copy so nested config values cannot be mutated globally."""
    import copy
    return copy.deepcopy(config or {})


def split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    escape = False

    for ch in text:
        if escape:
            current.append(ch)
            escape = False
            continue
        if ch == "\\":
            current.append(ch)
            escape = True
            continue
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            current.append(ch)
            quote = ch
            continue
        if ch in "{[(":
            depth += 1
            current.append(ch)
            continue
        if ch in "}])":
            depth -= 1
            current.append(ch)
            continue
        if ch == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(ch)

    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def parse_config_value(raw_value: str):
    import json
    value = raw_value.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None

    if (
        (value.startswith("{") and value.endswith("}"))
        or (value.startswith("[") and value.endswith("]"))
        or (value.startswith('"') and value.endswith('"'))
    ):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]

    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def set_nested_config_value(config: dict, dotted_key: str, value) -> None:
    key_parts = [part.strip() for part in dotted_key.split(".") if part.strip()]
    if not key_parts:
        raise ValueError(f"Invalid empty config key in MODEL_NAME: {dotted_key!r}")

    cursor = config
    for part in key_parts[:-1]:
        existing = cursor.get(part)
        if existing is None:
            existing = {}
            cursor[part] = existing
        if not isinstance(existing, dict):
            raise ValueError(
                f"Cannot set nested MODEL_NAME config key {dotted_key!r}: "
                f"{part!r} is already set to a non-dict value."
            )
        cursor = existing
    cursor[key_parts[-1]] = value


def parse_inline_model_config(config_text: str) -> dict:
    config: dict = {}
    text = (config_text or "").strip()
    if not text:
        return config

    for assignment in split_top_level_commas(text):
        if "=" not in assignment:
            raise ValueError(
                "MODEL_NAME inline config entries must use key=value syntax. "
                f"Got: {assignment!r}"
            )
        key, raw_value = assignment.split("=", 1)
        set_nested_config_value(config, key.strip(), parse_config_value(raw_value))
    return config


def parse_model_name_and_inline_config(model_name: str | None) -> tuple[str, dict]:
    """Parse MODEL_NAME values like 'gpt-5.4(reasoning.effort=high)'."""
    model = (model_name or "").strip()

    # Backward compatibility with old values such as "gemini:gemini-...".
    if ":" in model:
        prefix, suffix = model.split(":", 1)
        if suffix.strip() and prefix.strip().lower() in {
            "gemini", "google", "openai", "deepseek", "anthropic", "together",
            "openrouter", "groq", "fireworks", "google_genai", "google-ai", "google_ai",
        }:
            model = suffix.strip()

    if model.endswith(")") and "(" in model:
        base, inline = model.rsplit("(", 1)
        base = base.strip()
        inline = inline[:-1].strip()
        return base, parse_inline_model_config(inline)

    return model, {}


def normalize_config_model_name(model_name: str | None) -> str:
    model, _inline_config = parse_model_name_and_inline_config(model_name)
    return model


def merge_configs(base: dict, extra: dict) -> dict:
    """Recursively merge two configuration dictionaries."""
    import copy
    merged = copy_config(base)
    for key, value in (extra or {}).items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def get_model_extra_config(api_provider: str | None, model_name: str | None) -> dict:
    """Return inline non-vanilla config embedded inside MODEL_NAME."""
    _provider = normalize_config_provider(api_provider)
    _model, inline_config = parse_model_name_and_inline_config(model_name)
    return copy_config(inline_config)


def get_provider_model_config(
    api_provider: str | None,
    provider_extra_configs: dict[str, dict] | None = None,
) -> dict:
    """Return deliberate provider-level non-default configuration additions."""
    provider = normalize_config_provider(api_provider)
    configs = provider_extra_configs or {}
    return copy_config(configs.get(provider, {}))


def get_effective_model_config(
    api_provider: str | None,
    model_name: str | None,
    provider_extra_configs: dict[str, dict] | None = None,
) -> dict:
    """Return final generation config: provider defaults first, inline model config on top."""
    provider_config = get_provider_model_config(api_provider, provider_extra_configs)
    model_config = get_model_extra_config(api_provider, model_name)
    return merge_configs(provider_config, model_config)


def flatten_config_for_label(config: dict, prefix: str = "") -> list[str]:
    items: list[str] = []
    for key in sorted(config):
        if key == "actual_model":
            continue
        value = config[key]
        label_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            items.extend(flatten_config_for_label(value, label_key))
        else:
            items.append(f"{label_key}={value}")
    return items


def get_model_display_name(api_provider: str | None, model_name: str | None) -> str:
    """Return the model name used in outputs/tables.

    The suffix describes only inline non-vanilla additions from MODEL_NAME.
    Provider-level Gemini defaults are intentionally omitted from the name.
    """
    model, inline_config = parse_model_name_and_inline_config(model_name)
    label_parts = flatten_config_for_label(inline_config)
    if not label_parts:
        return model
    return f"{model} (" + ", ".join(label_parts) + ")"


def vocabulary_hint(game) -> str:
    alphabet = getattr(game.dfa, "input_symbols", None)
    if not alphabet:
        return "vocabulary: unknown"
    return "DFA vocabulary: {" + ", ".join(sorted(alphabet)) + "}"


def get_tools():
    from tools import IsWordInLanguageTool, EvaluateDFACandidateTool

    return [
        IsWordInLanguageTool,
        EvaluateDFACandidateTool,
    ]
