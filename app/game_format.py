"""
Responsible for:

- Defining the interactive game configuration structure for DFA-learning sessions
- Managing tool selection, hint selection, and tool-to-game attachment
- Building the full game prompt presented to the model
- Constructing interactive game instances from selected tools and hints
- Serving as the configuration layer connecting DFA, prompts, tools, and runtime settings
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, TYPE_CHECKING, Type
from constants import HINTS, GAME_PROMPT, HINT_PROMPTS
from game_types import ToolInterface, EvaluationToolInterface
if TYPE_CHECKING:
    from dfa_class import MinimalDFA


@dataclass
class GameFormat:
    dfa: Optional["MinimalDFA"]
    tools: List[ToolInterface]
    hints: List[str]
    max_tool_calls: int = 0
    game_prompt: str = GAME_PROMPT

    def __post_init__(self) -> None:
        for tool in self.tools:
            setattr(tool, "game", self)

    def get_game_prompt(self) -> str:
        parts = [self.game_prompt.rstrip(), ""]

        if self.tools:
            parts.append("Tools:")
            for tool in self.tools:
                name = getattr(tool, "tool_name", tool.__class__.__name__)
                prompt = getattr(tool, "prompt", "")
                if prompt:
                    parts.append(f"\n[{name}]\n{prompt.rstrip()}\n")
                else:
                    parts.append(f"\n[{name}]\n")

        if self.hints:
            parts.append("Hints:")
            for hint_key in self.hints:
                hint_obj = HINT_PROMPTS.get(hint_key, "")
                text = hint_obj(self) if callable(hint_obj) else hint_obj
                if text:
                    parts.append(f"- {text}")
                else:
                    parts.append(f"- {hint_key}")

        return "\n".join(parts).strip() + "\n"

    @classmethod
    def interactive_build(
        cls,
        *,
        dfa: Optional["MinimalDFA"],
        tool_classes: List[Type[ToolInterface]],
        max_tool_calls: int = 0,
        ask: Callable[[Type[ToolInterface]], bool],
        ask_hint: Callable[[str, str], bool],
        game_prompt: str = GAME_PROMPT,
    ) -> "GameFormat":
        chosen_tools: List[ToolInterface] = []
        chosen_hints: List[str] = []
        picked_evaluation = False

        for hint_key, display in HINTS.items():
            if ask_hint(hint_key, display):
                chosen_hints.append(hint_key)

        for tool_cls in tool_classes:
            is_eval = issubclass(tool_cls, EvaluationToolInterface)
            if picked_evaluation and is_eval:
                continue

            if ask(tool_cls):
                chosen_tools.append(tool_cls())
                if is_eval:
                    picked_evaluation = True

        return cls(
            dfa=dfa,
            tools=chosen_tools,
            hints=chosen_hints,
            max_tool_calls=max_tool_calls,
            game_prompt=game_prompt,
        )