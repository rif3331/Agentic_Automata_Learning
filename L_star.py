"""
Responsible for:

- Implementing L* (LStar) DFA learning strategy using AALpy
- Managing membership and equivalence queries through teacher and oracle components
- Converting learned hypotheses into DFA representations for equivalence checking
- Tracking query counts, query history, caching, and counterexample generation
- Running L* experiments and producing DFA learning strategy results
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from collections import deque

from automata.fa.dfa import DFA
from aalpy.base import SUL
from aalpy.learning_algs import run_Lstar
from utils import (
    _word_to_seq,
    _word_repr,
)

from html_code.lstar_comparison_html import write_lstar_comparison_html



LSTAR_RUNTIME_NAME = "LSTAR"



def aalpy_to_dfa(hyp: Any, alphabet: List[Any]) -> DFA:
    init = getattr(hyp, "initial", None) or getattr(hyp, "initial_state", None)
    if init is None:
        raise TypeError("Hypothesis has no initial state")

    q = deque([init])
    seen = {init}
    states_list = [init]

    while q:
        s = q.popleft()
        trans = getattr(s, "transitions", {})
        if not isinstance(trans, dict):
            raise TypeError("Hypothesis state transitions are not a dict")

        for a in alphabet:
            ns = trans.get(a, None)
            if ns is None:
                continue

            if ns not in seen:
                seen.add(ns)
                q.append(ns)
                states_list.append(ns)

    id_of = {st: f"q{i}" for i, st in enumerate(states_list)}
    states = set(id_of.values())
    transitions = {}
    final_states = set()

    for st in states_list:
        sid = id_of[st]
        trans = getattr(st, "transitions", {})
        nt = {}

        for a in alphabet:
            ns = trans.get(a, None)
            if ns is not None:
                nt[a] = id_of[ns]

        transitions[sid] = nt

        acc = False
        if hasattr(st, "is_accepting"):
            acc = bool(getattr(st, "is_accepting"))
        elif hasattr(st, "accepting"):
            acc = bool(getattr(st, "accepting"))

        if acc:
            final_states.add(sid)

    return DFA(
        states=states,
        input_symbols=set(alphabet),
        transitions=transitions,
        initial_state=id_of[init],
        final_states=final_states,
        allow_partial=True,
    )


class QueryLogItem(tuple):
    __slots__ = ()

    def __new__(
        cls,
        kind: str,
        word: Optional[str],
        result: Any,
        link: Optional[str] = None,
        automaton: Any = None,
    ):
        if kind == "EQ" and result is True:
            word = None
        if automaton is None:
            return tuple.__new__(cls, (kind, word, result, link))
        return tuple.__new__(cls, (kind, word, result, link, automaton))

    @property
    def kind(self) -> str:
        return self[0]

    @property
    def word(self) -> Optional[str]:
        return self[1]

    @property
    def result(self):
        return self[2]

    @property
    def link(self) -> Optional[str]:
        return self[3]


class DFATeacherSUL(SUL):
    def __init__(self, dfa: DFA):
        super().__init__()
        self.dfa = dfa
        self.state = None

    def pre(self):
        self.state = self.dfa.initial_state

    def post(self):
        return

    def step(self, inp):
        if self.state is None:
            return False

        trans = self.dfa.transitions.get(self.state, {})
        if inp not in trans:
            self.state = None
            return False

        self.state = trans[inp]
        return self.state in self.dfa.final_states

    def accepts_word(self, word: Sequence[Any]) -> bool:
        st = self.dfa.initial_state

        if not word:
            return st in self.dfa.final_states

        for a in word:
            trans = self.dfa.transitions.get(st, {})
            if a not in trans:
                return False
            st = trans[a]

        return st in self.dfa.final_states

    def query(self, word: Sequence[Any]):
        mem = bool(self.accepts_word(word))
        return [mem]


class CountingSUL(SUL):
    def __init__(
        self,
        sul: DFATeacherSUL,
        history: Optional[List[Any]] = None,
        knowledge_state: Optional[dict[str, Any]] = None,
        dfa: Optional[DFA] = None,
        strategy_name: str = LSTAR_RUNTIME_NAME,
    ):
        super().__init__()
        self.sul = sul
        self.mq_queries = 0
        self.history = history
        self.cache: Dict[tuple[Any, ...], bool] = {}
        self.printed_mq_queries: set[tuple[Any, ...]] = set()
        self.knowledge_state = knowledge_state if knowledge_state is not None else {}
        self.dfa = dfa
        self.strategy_name = strategy_name

    def pre(self):
        return self.sul.pre()

    def post(self):
        return self.sul.post()

    def step(self, inp):
        return self.sul.step(inp)

    def query(self, word: Sequence[Any]):
        key = tuple(word) if word is not None else tuple()
        w = _word_repr(list(key))

        if key in self.cache:
            return [self.cache[key]]

        mem = bool(self.sul.accepts_word(list(key)))

        self.cache[key] = mem
        self.mq_queries += 1

        if self.history is not None:
            self.history.append(QueryLogItem("MQ", w, mem, None))

        return [mem]

    def membership(self, word: Sequence[Any]) -> bool:
        key = tuple(word) if word is not None else tuple()

        if key in self.cache:
            return self.cache[key]

        out = self.query(list(key))
        return bool(out[-1])


class MinimalDFAEqOracle:
    def __init__(
        self,
        target_dfa: DFA,
        alphabet: List[Any],
        history: Optional[List[Any]] = None,
        knowledge_state: Optional[dict[str, Any]] = None,
        strategy_name: str = LSTAR_RUNTIME_NAME,
        minimal_counterexample: bool = False,
        counterexample_max_extra_len: int = 3,
    ):
        self.alphabet = sorted(alphabet, key=str)

        from dfa_class import MinimalDFA

        self.target_dfa = target_dfa
        self.knowledge_state = knowledge_state if knowledge_state is not None else {}
        self.strategy_name = strategy_name

        self.minimal_counterexample = bool(minimal_counterexample)
        self.counterexample_max_extra_len = int(counterexample_max_extra_len)
        raw_counterexample_mode = int(getattr(target_dfa, "counterexample_mode", 0))
        self.counterexample_mode = 0 if raw_counterexample_mode == 2 else raw_counterexample_mode

        self.target_min = MinimalDFA.from_dfa(
            target_dfa,
            run_strategy=False,
            minimal_counterexample=self.minimal_counterexample,
            counterexample_max_extra_len=self.counterexample_max_extra_len,
            counterexample_mode=self.counterexample_mode,
        )

        self.num_queries = 0
        self.num_steps = 0
        self.history = history

    def reset(self):
        return

    def find_cex(self, hypothesis):
        self.num_queries += 1
        self.num_steps += 1

        hyp_dfa = aalpy_to_dfa(hypothesis, self.alphabet)

        from dfa_class import MinimalDFA

        hyp_min = MinimalDFA.from_dfa(
            hyp_dfa,
            run_strategy=False,
            minimal_counterexample=self.minimal_counterexample,
            counterexample_max_extra_len=self.counterexample_max_extra_len,
            counterexample_mode=self.counterexample_mode,
        )
        ok, w = self.target_min.eq(hyp_min)

        seq = _word_to_seq(w, self.alphabet)
        cex = _word_repr(seq) if not ok else None

        link = write_lstar_comparison_html(
            self.target_dfa,
            hyp_dfa,
            ok=bool(ok),
            counterexample=cex,
        )

        if ok:
            if self.history is not None:
                self.history.append(QueryLogItem("EQ", None, True, link, hyp_dfa))
            return None

        if self.history is not None:
            self.history.append(QueryLogItem("EQ", cex, False, link, hyp_dfa))

        return seq


class LStarStrategy:
    def __init__(self, print_level: int = 0):
        self.print_level = print_level

    def run(self, dfa):

        from game_types import DFAStrategy

        history: List[Any] = []

        alphabet = sorted(dfa.input_symbols, key=str)

        teacher = DFATeacherSUL(dfa)

        sul = CountingSUL(
            teacher,
            history=history,
        )

        eqo = MinimalDFAEqOracle(
            dfa,
            alphabet,
            history=history,
            minimal_counterexample=bool(getattr(dfa, "minimal_counterexample", False)),
            counterexample_max_extra_len=int(getattr(dfa, "counterexample_max_extra_len", 3)),
        )

        run_Lstar(
            alphabet,
            sul,
            eqo,
            automaton_type="dfa",
            print_level=0,
            cache_and_non_det_check=False,
        )

        eq_q = eqo.num_queries
        mq_q = sul.mq_queries
        total = eq_q + mq_q

        return DFAStrategy.Result(
            total_queries=total,
            eq_queries=eq_q,
            mq_queries=mq_q,
            history=history,
        )