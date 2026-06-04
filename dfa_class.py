"""
Responsible for:

- Defining the core MinimalDFA structure used throughout the project
- Managing DFA construction, minimization, equivalence checking, and membership queries
- Generating counterexamples and deterministic witnesses for DFA comparison
- Running learning strategies and storing strategy performance results
- Supporting DFA visualization, normalization, and random-pair comparison utilities
"""
from __future__ import annotations
import random
import math
import hashlib
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from automata.fa.dfa import DFA
from game_types import DFAStrategy, FormalStructure
from html_code.draw_DFA_html import draw_DFA_html, draw_DFA_html_option2
from constants import STRATEGIES


def _make_total(
    *,
    states: Set[Any],
    input_symbols: Set[str],
    transitions: Dict[Any, Dict[str, Any]],
) -> tuple[Set[Any], Dict[Any, Dict[str, Any]]]:
    states2: Set[Any] = set(states)
    trans2: Dict[Any, Dict[str, Any]] = {s: dict(transitions.get(s, {})) for s in states2}

    need_sink = False
    for s in states2:
        for a in input_symbols:
            if a not in trans2[s]:
                need_sink = True
                break
        if need_sink:
            break

    if not need_sink:
        return states2, trans2

    base_name = "__sink__"
    sink = base_name
    i = 0
    while sink in states2:
        i += 1
        sink = f"{base_name}{i}"

    states2.add(sink)
    trans2.setdefault(sink, {})

    for a in input_symbols:
        trans2[sink][a] = sink

    for s in states2:
        trans2.setdefault(s, {})
        for a in input_symbols:
            if a not in trans2[s]:
                trans2[s][a] = sink

    return states2, trans2




def _normalize_word(word: Any, input_symbols: Set[str]) -> Tuple[str, ...]:
    if word is None:
        return ()
    if isinstance(word, tuple):
        return tuple(word)
    if isinstance(word, list):
        return tuple(word)
    if isinstance(word, str):
        s = word.strip()
        if s == "":
            return ()
        if s in input_symbols:
            return (s,)
        if " " in s:
            parts = [p for p in s.split(" ") if p]
            return tuple(parts)
        if "," in s:
            parts = [p for p in s.split(",") if p]
            return tuple(parts)
        return (s,)
    try:
        return tuple(word)
    except TypeError:
        return (str(word),)


def _stable_dfa_fingerprint(dfa: DFA) -> str:
    alphabet = sorted(dfa.input_symbols, key=str)

    state_id = {}
    queue = deque([dfa.initial_state])
    state_id[dfa.initial_state] = 0

    while queue:
        s = queue.popleft()
        for a in alphabet:
            t = dfa.transitions[s][a]
            if t not in state_id:
                state_id[t] = len(state_id)
                queue.append(t)

    # safety: include unreachable states deterministically, if any
    for s in sorted(dfa.states, key=str):
        if s not in state_id:
            state_id[s] = len(state_id)

    parts = []
    parts.append("alphabet:" + "|".join(map(str, alphabet)))
    parts.append("initial:0")

    finals = sorted(state_id[s] for s in dfa.final_states)
    parts.append("finals:" + "|".join(map(str, finals)))

    trans_parts = []
    for s in sorted(state_id.keys(), key=lambda x: state_id[x]):
        sid = state_id[s]
        for a in alphabet:
            t = dfa.transitions[s][a]
            trans_parts.append(f"{sid} --{a}--> {state_id[t]}")

    parts.append("trans:" + "|".join(trans_parts))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _stable_pair_seed(dfa1: DFA, dfa2: DFA) -> int:
    f1 = _stable_dfa_fingerprint(dfa1)
    f2 = _stable_dfa_fingerprint(dfa2)
    pair = "||".join(sorted([f1, f2]))
    h = hashlib.sha256(pair.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def _pick_deterministic_short_cex(
    diff: DFA,
    *,
    seed: int,
    max_extra_len: int = 3,
    max_per_len: int = 200,
):
    k_min = diff.minimum_word_length()
    candidates = []
    for k in range(k_min, k_min + max_extra_len + 1):
        taken = 0
        for w in sorted(diff.words_of_length(k), key=lambda x: tuple(map(str, x))):
            candidates.append(w)
            taken += 1
            if taken >= max_per_len:
                break
    if not candidates:
        return None
    
    
    
    rng = random.Random(seed)
    return rng.choice(candidates)


class MinimalDFA(DFA, FormalStructure):
    def __init__(
        self,
        *,
        states,
        input_symbols,
        transitions,
        initial_state,
        final_states,
        allow_partial: bool = True,
        strategy_results: Optional[List[Dict[str, DFAStrategy.Result]]] = None,
        minimize: bool = True,
        retain_names: bool = False,
        make_total: bool = True,
        grid: bool = False,
        check_locality: bool = False,
        search_ngram_approx: bool = False,
        minimal_counterexample: bool = False,
        counterexample_max_extra_len: int = 3,
        counterexample_mode: int = 0,
    ):
        tmp = DFA(
            states=set(states),
            input_symbols=set(input_symbols),
            transitions=dict(transitions),
            initial_state=initial_state,
            final_states=set(final_states),
            allow_partial=allow_partial,
        )

        if make_total:
            states2, trans2 = _make_total(
                states=set(tmp.states),
                input_symbols=set(tmp.input_symbols),
                transitions=dict(tmp.transitions),
            )
            tmp = DFA(
                states=states2,
                input_symbols=set(tmp.input_symbols),
                transitions=trans2,
                initial_state=tmp.initial_state,
                final_states=set(tmp.final_states),
                allow_partial=False,
            )

        if minimize:
            tmp = tmp.minify(retain_names=retain_names)

        super().__init__(
            states=set(tmp.states),
            input_symbols=set(tmp.input_symbols),
            transitions=dict(tmp.transitions),
            initial_state=tmp.initial_state,
            final_states=set(tmp.final_states),
            allow_partial=False,
        )

        if strategy_results is None:
            strategy_results = []
        object.__setattr__(self, "strategy_results", strategy_results)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "check_locality", check_locality)

       

        object.__setattr__(self, "search_ngram_approx", search_ngram_approx)
        object.__setattr__(self, "ngram_approx_n", None)
        object.__setattr__(self, "ngram_approx_accs", {})

        object.__setattr__(self, "minimal_counterexample", bool(minimal_counterexample))
        object.__setattr__(self, "counterexample_max_extra_len", max(0, int(counterexample_max_extra_len)))
        object.__setattr__(self, "counterexample_mode", int(counterexample_mode))
        

    @classmethod
    def from_dfa(
        cls,
        dfa: DFA,
        run_strategy: bool = False,
        *,
        minimize: bool = True,
        retain_names: bool = False,
        make_total: bool = True,
        grid: bool = False,
        check_locality: bool = False,
        search_ngram_approx: bool = False,
        minimal_counterexample: bool = False,
        counterexample_max_extra_len: int = 3,
        counterexample_mode: int = 0,
    ) -> "MinimalDFA":
        inst = cls(
            states=dfa.states,
            input_symbols=dfa.input_symbols,
            transitions=dfa.transitions,
            initial_state=dfa.initial_state,
            final_states=dfa.final_states,
            allow_partial=getattr(dfa, "allow_partial", True),
            strategy_results=[],
            minimize=minimize,
            retain_names=retain_names,
            make_total=make_total,
            grid=grid,
            check_locality=check_locality,
            search_ngram_approx=search_ngram_approx,
            minimal_counterexample=minimal_counterexample,
            counterexample_max_extra_len=counterexample_max_extra_len,
            counterexample_mode=counterexample_mode,
        )
        if run_strategy:
            strategies = [s() for s in STRATEGIES]
            inst.run_strategies(strategies)
        return inst

    @classmethod
    def from_params(
        cls,
        *,
        states,
        input_symbols,
        transitions,
        initial_state,
        final_states,
        allow_partial: bool = True,
        minimize: bool = True,
        retain_names: bool = False,
        make_total: bool = True,
        grid: bool = False,
        check_locality: bool = False,
        search_ngram_approx: bool = False,
        minimal_counterexample: bool = False,
        counterexample_max_extra_len: int = 3,
        counterexample_mode: int = 0,
    ) -> "MinimalDFA":
        tmp = DFA(
            states=states,
            input_symbols=input_symbols,
            transitions=transitions,
            initial_state=initial_state,
            final_states=final_states,
            allow_partial=allow_partial,
        )
        return cls.from_dfa(
            tmp,
            run_strategy=False,
            minimize=minimize,
            retain_names=retain_names,
            make_total=make_total,
            grid=grid,
            check_locality=check_locality,
            search_ngram_approx=search_ngram_approx,
            minimal_counterexample=minimal_counterexample,
            counterexample_max_extra_len=counterexample_max_extra_len,
            counterexample_mode=counterexample_mode,
        )

    def run_strategies(
        self,
        strategies: Iterable[DFAStrategy],
    ) -> Dict[str, Dict[str, DFAStrategy.Result]]:
        runs: List[Dict[str, DFAStrategy.Result]] = getattr(self, "strategy_results", [])
        if runs is None:
            runs = []
            object.__setattr__(self, "strategy_results", runs)

        by_strategy: Dict[str, DFAStrategy.Result] = {}
        name_counts: Dict[str, int] = {}

        for strat in strategies:
            res = strat.run(self)

            base = type(strat).__name__
            if base in name_counts:
                name_counts[base] += 1
                key = f"{base}#{name_counts[base]}"
            else:
                name_counts[base] = 1
                key = base

            by_strategy[key] = res

        runs.append(by_strategy)
        return {"by_strategy": by_strategy}

    def mq(self, word: str) -> bool:
        w = _normalize_word(word, set(self.input_symbols))
        return bool(self.accepts_input(w))

    def eq(self, other: object) -> Tuple[bool, Optional[str]]:
        if not isinstance(other, MinimalDFA):
            return (False, None)

        diff = self.symmetric_difference(other, minify=True)
        if diff.isempty():
            return (True, None)

        use_minimal = bool(getattr(self, "minimal_counterexample", False))
        max_extra_len = int(getattr(self, "counterexample_max_extra_len", 3))

        if use_minimal:
            k = diff.minimum_word_length()
            w = next(diff.words_of_length(k))
        else:
            seed = _stable_pair_seed(self, other)
            w = _pick_deterministic_short_cex(
                diff,
                seed=seed,
                max_extra_len=max_extra_len,
                max_per_len=20000,
            )
            if w is None:
                k = diff.minimum_word_length()
                w = next(diff.words_of_length(k))

        if isinstance(w, tuple):
            return (False, " ".join(w))
        return (False, str(w))

    def draw(self) -> str:
        if getattr(self, "grid", True):
            return draw_DFA_html_option2(self)
        return draw_DFA_html(self)