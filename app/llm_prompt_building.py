"""
Responsible for:

- Building prompts and follow-up messages sent from the runtime to the model
- Formatting initial prompts, tool results, and knowledge-state payloads
- Managing remaining tool-call information inside prompts
- Constructing counterexample context blocks for failed DFA guesses
- Supporting prompt serialization, JSON formatting, and model-input preparation
"""
from __future__ import annotations
import json
from typing import Any


def json_default(_owner: Any, o: Any) -> Any:
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def build_initial_text(owner: Any) -> str:
    prompt = owner.game.get_game_prompt()
    max_calls = getattr(owner.game, "max_tool_calls", 0)

    if "{MAX_CALLS}" in prompt:
        prompt = prompt.replace("{MAX_CALLS}", str(max_calls))

    owner._last_initial_prompt = prompt
    return prompt


def print_sent_to_model(owner: Any, text: str, *, step: int | None = None, tag: str = "") -> None:
    if tag == "CHAT_INIT":
        label = "Initial prompt sent to model"
        icon = "📥"
    elif tag == "CHAT_TOOL_RESULT":
        call_number = getattr(owner, "_last_oracle_response_call_number", None)
        if isinstance(call_number, int) and call_number > 0:
            label = f"Oracle response to tool call #{call_number} sent to model"
        else:
            label = "Oracle response sent to model"
        icon = "🔮"
    else:
        label = "Input sent to model"
        icon = "📥"

    print("\n" + "!" * 90)
    print(f"{icon} {label}")
    print("!" * 90)
    print(text)
    print("!" * 90 + "\n")

def get_initial_prompt_with_remaining_calls(owner: Any) -> str:
    prompt = owner._last_initial_prompt or owner.game.get_game_prompt()
    max_calls = getattr(owner.game, "max_tool_calls", 0)

    if not isinstance(max_calls, int) or max_calls <= 0:
        return prompt

    remaining = max(0, max_calls - owner._call_counter)
    if "{MAX_CALLS}" in prompt:
        return prompt.replace("{MAX_CALLS}", str(remaining))

    return prompt


def build_last_bad_eq_block(owner: Any) -> str:
    if not isinstance(owner.eq_dfa_guesses, list) or not owner.eq_dfa_guesses:
        return ""

    _step, cand_obj, witness = owner.eq_dfa_guesses[-1]
    if not isinstance(witness, str) or not witness.strip():
        return ""

    payload = {
        "candidate_dfa": cand_obj,
        "counterexample": witness,
    }

    return (
        "<LAST_BAD_EQ>\n"
        + json.dumps(payload, ensure_ascii=False, default=owner._json_default)
        + "\n</LAST_BAD_EQ>\n\n"
    )


def strip_knowledge_state_deep(owner: Any, obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("knowledge_state", "html"):
                continue
            out[k] = strip_knowledge_state_deep(owner, v)
        return out

    if isinstance(obj, list):
        return [strip_knowledge_state_deep(owner, x) for x in obj]

    return obj


def build_model_text(owner: Any, *, include_knowledge_state: bool = True) -> str:
    prompt = owner.get_initial_prompt_with_remaining_calls()
    last_bad_eq_block = owner._build_last_bad_eq_block()

    if include_knowledge_state:
        ks_payload = owner._snapshot_knowledge_state(owner.current_knowledge_state)
        base = (
            prompt.rstrip()
            + "\n\n<KNOWLEDGE_STATE>\n"
            + json.dumps(ks_payload, ensure_ascii=False, default=owner._json_default)
            + "\n</KNOWLEDGE_STATE>\n\n"
            + last_bad_eq_block
        )
    else:
        base = prompt.rstrip() + "\n"

    if owner.pending_tool_result is None:
        return base

    cleaned = owner._strip_knowledge_state_deep(owner.pending_tool_result)
    return (
        base.rstrip()
        + "\n\n<TOOL_RESULT>\n"
        + json.dumps(cleaned, ensure_ascii=False, default=owner._json_default)
        + "\n</TOOL_RESULT>\n"
    )


def build_followup_text(owner: Any, tool_reply: Any) -> str:
    cleaned = owner._strip_knowledge_state_deep(tool_reply)
    return f"""
    <TOOL_RESULT>
    {json.dumps(cleaned, ensure_ascii=False, default=owner._json_default)}
    </TOOL_RESULT>
    """