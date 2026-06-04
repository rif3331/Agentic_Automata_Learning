"""
Responsible for:

- Parsing and extracting text, thoughts, and reasoning from model responses
- Handling response normalization across OpenAI and Gemini providers
- Repairing and parsing tool-action JSON embedded in model output
- Formatting message histories and converting messages into chat model formats
- Extracting tool calls and processing tool-related content from raw assistant responses
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional, Tuple
JsonDict = Dict[str, Any]
_TOOL_ACTION_RE = re.compile(r"<TOOL_ACTION>\s*(.*?)\s*</TOOL_ACTION>", re.DOTALL)


def extract_openai_text_and_thoughts(owner: Any, resp: Any) -> Tuple[str, str]:
    text_parts: List[str] = []
    thought_parts: List[str] = []

    content = getattr(resp, "content", None) or []

    if isinstance(content, str):
        return content, ""

    if not isinstance(content, list):
        return "", ""

    for item in content:
        if isinstance(item, dict):
            item_type = item.get("type")
            item_text = item.get("text")
            item_summary = item.get("summary")
        else:
            item_type = getattr(item, "type", None)
            item_text = getattr(item, "text", None)
            item_summary = getattr(item, "summary", None)

        if item_type == "text":
            if isinstance(item_text, str) and item_text.strip():
                text_parts.append(item_text)

        elif item_type == "reasoning":
            summary = item_summary or []

            if isinstance(summary, list):
                for s in summary:
                    if isinstance(s, dict):
                        s_text = s.get("text")
                    else:
                        s_text = getattr(s, "text", None) if s is not None else None

                    if isinstance(s_text, str) and s_text.strip():
                        thought_parts.append(s_text)

    return "\n".join(text_parts).strip(), "\n".join(thought_parts).strip()


def extract_gemini_text_and_thoughts(owner: Any, resp: Any) -> Tuple[str, str]:
    text_parts: List[str] = []
    thought_parts: List[str] = []

    candidates = getattr(resp, "candidates", None) or []
    if candidates:
        cand0 = candidates[0]
        content = getattr(cand0, "content", None)
        parts = getattr(content, "parts", None) or []

        for part in parts:
            part_text = getattr(part, "text", None)
            if not isinstance(part_text, str) or not part_text.strip():
                continue

            if bool(getattr(part, "thought", False)):
                thought_parts.append(part_text)
            else:
                text_parts.append(part_text)

    full_text = "\n".join(text_parts).strip()
    full_thoughts = "\n".join(thought_parts).strip()

    if not full_text:
        full_text = getattr(resp, "text", "") or ""

    return full_text, full_thoughts


def coerce_text(owner: Any, text) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        return "\n".join(str(x) for x in text)
    return str(text)


def print_raw_model_text(owner: Any, text: str, *, tag: str = "", step: int | None = None) -> None:
    if not owner._print_raw_model_output:
        return

    # This is the model output that contains the agent's requested tool call.
    # `owner` here is the model router, not the game runtime, so it does not
    # own the real oracle budget counter. Use the conversation/model step passed
    # from llm_runtime for display only. This prevents the heading from resetting
    # to #1 after oracle errors such as NO_TOOL_CALLS_DETECTED.
    call_number = step if isinstance(step, int) and step > 0 else None
    call_part = f" #{call_number}" if isinstance(call_number, int) else ""
    head = f"🤖 Agent tool call{call_part}"
    print("\n" + "=" * 60)
    print(head)
    print("=" * 60)
    print(str(text or "")[:4000] + ("..." if text and len(str(text)) > 4000 else ""))
    print("=" * 60 + "\n")


def print_model_raw(owner: Any, resp: Any) -> None:
    return

def repair_tool_json(owner: Any, s: str) -> str:
    s = (s or "").strip()

    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)

    first = s.find("{")
    last = s.rfind("}")
    if first != -1:
        s = s[first:] if last == -1 else s[first:last + 1]

    open_curly = s.count("{")
    close_curly = s.count("}")
    if close_curly < open_curly:
        s += "}" * (open_curly - close_curly)

    open_square = s.count("[")
    close_square = s.count("]")
    if close_square < open_square:
        s += "]" * (open_square - close_square)

    s = re.sub(r",\s*([}\]])", r"\1", s)

    return s


def safe_json_loads(owner: Any, s: str) -> Optional[JsonDict]:
    raw = (s or "").strip()

    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    try:
        repaired = owner._repair_tool_json(raw)
        obj = json.loads(repaired)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def strip_knowledge_state_from_tool_result(owner: Any, text: str) -> str:
    if "<TOOL_RESULT>" not in text:
        return text
    try:
        m = re.search(r"<TOOL_RESULT>\s*(\{.*\})\s*</TOOL_RESULT>", text, re.DOTALL)
        if not m:
            return text
        data = json.loads(m.group(1))
        if isinstance(data, dict):
            for out in data.get("tool_outputs", []):
                if isinstance(out, dict):
                    out.pop("knowledge_state", None)
        return f"<TOOL_RESULT>\n{json.dumps(data, ensure_ascii=False)}\n</TOOL_RESULT>"
    except Exception:
        return text


def messages_to_string(owner: Any, messages: List[Any]) -> str:
    parts = []
    for msg in messages:
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        content = getattr(msg, "content", "") or ""
        if role == "human" or "Human" in str(role):
            parts.append("USER:\n" + content)
        else:
            parts.append("ASSISTANT:\n" + content)
    return "\n\n".join(parts).strip()


def messages_to_openai_format(owner: Any, messages: List[Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if owner.system_prompt:
        out.append({"role": "system", "content": owner.system_prompt})

    for msg in messages:
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        content = getattr(msg, "content", "") or ""
        if role == "human" or "Human" in str(role):
            out.append({"role": "user", "content": content})
        else:
            out.append({"role": "assistant", "content": content})
    return out


def from_text(owner: Any, text: str) -> JsonDict:
    if (
        "<TOOL_ACTION>" in (text or "")
        and owner._print_tool_io
    ):
        if not (owner._print_tool_action_once_per_step and owner._last_send_was_tool_result):
            print("\n" + "=" * 60)
            print("🧠 MODEL (ASSISTANT TOOL_ACTION):")
            print(text[:1000])
            print("=" * 60 + "\n")
    text = owner._coerce_text(text)
    calls = owner._extract_tool_actions_from_text(text)
    return {"tool_calls": calls, "content": text} if calls else {"content": text}


def extract_tool_actions_from_text(owner: Any, text: str) -> List[JsonDict]:
    out = []
    text = owner._coerce_text(text)
    for m in _TOOL_ACTION_RE.finditer(text or ""):
        raw_json = m.group(1)
        obj = owner._safe_json_loads(raw_json)
        if isinstance(obj, dict) and isinstance(obj.get("tool_name"), str):
            out.append({"name": obj["tool_name"], "args": obj.get("input", {})})
    return out