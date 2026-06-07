"""
Responsible for:

- Defining shared protocols, typed structures, and abstract interfaces for the project
- Describing DFA strategies, formal structures, and tool interaction contracts
- Providing typed request/response schemas for tool execution and knowledge state exchange
- Establishing the base interfaces for standard tools and evaluation tools
- Serving as the common type-definition layer used across runtime, tools, and strategies
"""
from __future__ import annotations
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    runtime_checkable,
)
from abc import ABC, abstractmethod
JsonDict = Dict[str, Any]
QueryLogItem = Tuple[str, Any]
KnowledgeState = Dict[str, set[str]]


class DFAStrategy(Protocol):
    class Result(NamedTuple):
        total_queries: int
        eq_queries: int
        mq_queries: int
        history: List[QueryLogItem]

    def run(self, dfa: "FormalStructure") -> "DFAStrategy.Result":
        ...


@runtime_checkable
class FormalStructure(Protocol):
    def draw(self) -> str:
        ...

    def run_strategies(self, strategies: Iterable[DFAStrategy]) -> List[Any]:
        ...

    def mq(self, word: str) -> bool:
        ...

    def eq(self, other: object) -> Tuple[bool, Optional[str]]:
        ...


class ToolInput(TypedDict):
    tool_name: str
    call_count: int
    input: Dict[str, Any]
    knowledge_state: KnowledgeState


class ToolOutput(TypedDict):
    tool_name: str
    call_count: int
    error: Optional[str]
    output: Optional[Dict[str, Any]]
    knowledge_state: KnowledgeState


class ToolInterface(ABC):
    prompt: str
    tool_name: str

    @abstractmethod
    def invoke(self, request: ToolInput) -> ToolOutput:
        pass


class EvaluationToolInterface(ToolInterface, ABC):
    @abstractmethod
    def draw(self) -> str:
        pass

    @abstractmethod
    def invoke(self, request: ToolInput) -> ToolOutput:
        pass


class EvaluationToolOutput(TypedDict):
    optimal: bool
    score: float