"""
Responsible for:

- Generating random accessible DFAs for experiments
- Sampling DFA structures using Boltzmann-based combinatorial generation
- Building alphabets, partitions, and transition structures for random automata
- Constructing minimized DFA objects from generated transition systems
- Supporting randomized DFA creation used as targets in learning experiments
"""
from __future__ import annotations
import math
import random
from typing import Dict, List, Set, Tuple, Optional
from dfa_class import MinimalDFA


def make_random_dfa(
    n_states: int,
    alphabet_size: int,
    seed: Optional[int] = None,
) -> MinimalDFA:
    base = boltzmann_sample_accessible_dfa(
            n_states,alphabet_size,seed=seed
        )
    return MinimalDFA.from_dfa(base, run_strategy=True)


# ---------------- Regex -> DFA support ----------------
# This is intentionally implemented here (instead of a new file), because the
# launcher/demo should be updated by replacing existing files only.

_REGEX_EPSILON = "ε"
_REGEX_CONCAT = "·"


def _regex_tokenize(regex: str) -> list[str]:
    """Tokenize a small regular-expression syntax used by the demo.

    Supported syntax:
      - symbols: single non-operator characters such as a, b, 0, 1
      - union: |
      - implicit concatenation: ab or a(b|c)
      - Kleene star: *
      - plus: +
      - optional: ?
      - parentheses: ( )
      - epsilon: ε, eps, epsilon

    Spaces are ignored.
    """
    text = (regex or "").strip()
    if not text:
        raise ValueError("Regular expression is empty")

    tokens: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if text.startswith("epsilon", i):
            tokens.append(_REGEX_EPSILON)
            i += len("epsilon")
            continue
        if text.startswith("eps", i):
            tokens.append(_REGEX_EPSILON)
            i += len("eps")
            continue
        if ch in {"ε", "|", "*", "+", "?", "(", ")"}:
            tokens.append(ch)
            i += 1
            continue
        if ch == "\\":
            if i + 1 >= len(text):
                raise ValueError("Backslash at end of regular expression")
            tokens.append(text[i + 1])
            i += 2
            continue
        # Demo alphabet symbols are character-based, matching the rest of the
        # project UI where words are typed as strings such as abba.
        tokens.append(ch)
        i += 1
    return tokens


def _regex_needs_concat(left: str, right: str) -> bool:
    left_is_atom_end = left not in {"|", "("}
    right_is_atom_start = right not in {"|", ")", "*", "+", "?"}
    return left_is_atom_end and right_is_atom_start


def _regex_add_concat(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        if out and _regex_needs_concat(out[-1], tok):
            out.append(_REGEX_CONCAT)
        out.append(tok)
    return out


def _regex_to_postfix(regex: str) -> list[str]:
    tokens = _regex_add_concat(_regex_tokenize(regex))
    prec = {"|": 1, _REGEX_CONCAT: 2, "*": 3, "+": 3, "?": 3}
    postfix: list[str] = []
    stack: list[str] = []

    for tok in tokens:
        if tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                postfix.append(stack.pop())
            if not stack:
                raise ValueError("Mismatched parentheses in regular expression")
            stack.pop()
        elif tok in prec:
            # Unary postfix operators bind immediately to the previous atom.
            if tok in {"*", "+", "?"}:
                postfix.append(tok)
            else:
                while stack and stack[-1] != "(" and prec.get(stack[-1], 0) >= prec[tok]:
                    postfix.append(stack.pop())
                stack.append(tok)
        else:
            postfix.append(tok)

    while stack:
        op = stack.pop()
        if op == "(":
            raise ValueError("Mismatched parentheses in regular expression")
        postfix.append(op)
    return postfix


def _regex_postfix_to_nfa(postfix: list[str]):
    epsilon = None
    next_state = 0
    transitions: dict[int, dict[str | None, set[int]]] = {}
    alphabet: set[str] = set()

    def new_state() -> int:
        nonlocal next_state
        s = next_state
        next_state += 1
        transitions.setdefault(s, {})
        return s

    def add_edge(src: int, sym: str | None, dst: int) -> None:
        transitions.setdefault(src, {}).setdefault(sym, set()).add(dst)
        transitions.setdefault(dst, {})

    stack: list[tuple[int, int]] = []

    for tok in postfix:
        if tok == _REGEX_CONCAT:
            if len(stack) < 2:
                raise ValueError("Invalid concatenation in regular expression")
            s2, e2 = stack.pop()
            s1, e1 = stack.pop()
            add_edge(e1, epsilon, s2)
            stack.append((s1, e2))
        elif tok == "|":
            if len(stack) < 2:
                raise ValueError("Invalid union in regular expression")
            s2, e2 = stack.pop()
            s1, e1 = stack.pop()
            s = new_state()
            e = new_state()
            add_edge(s, epsilon, s1)
            add_edge(s, epsilon, s2)
            add_edge(e1, epsilon, e)
            add_edge(e2, epsilon, e)
            stack.append((s, e))
        elif tok == "*":
            if not stack:
                raise ValueError("Invalid Kleene star in regular expression")
            s1, e1 = stack.pop()
            s = new_state()
            e = new_state()
            add_edge(s, epsilon, s1)
            add_edge(s, epsilon, e)
            add_edge(e1, epsilon, s1)
            add_edge(e1, epsilon, e)
            stack.append((s, e))
        elif tok == "+":
            if not stack:
                raise ValueError("Invalid plus operator in regular expression")
            s1, e1 = stack.pop()
            s = new_state()
            e = new_state()
            add_edge(s, epsilon, s1)
            add_edge(e1, epsilon, s1)
            add_edge(e1, epsilon, e)
            stack.append((s, e))
        elif tok == "?":
            if not stack:
                raise ValueError("Invalid optional operator in regular expression")
            s1, e1 = stack.pop()
            s = new_state()
            e = new_state()
            add_edge(s, epsilon, s1)
            add_edge(s, epsilon, e)
            add_edge(e1, epsilon, e)
            stack.append((s, e))
        else:
            s = new_state()
            e = new_state()
            if tok == _REGEX_EPSILON:
                add_edge(s, epsilon, e)
            else:
                alphabet.add(tok)
                add_edge(s, tok, e)
            stack.append((s, e))

    if len(stack) != 1:
        raise ValueError("Invalid regular expression")

    start, accept = stack[0]
    return start, accept, transitions, alphabet


def _epsilon_closure(states: set[int], transitions: dict[int, dict[str | None, set[int]]]) -> frozenset[int]:
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for dst in transitions.get(s, {}).get(None, set()):
            if dst not in closure:
                closure.add(dst)
                stack.append(dst)
    return frozenset(closure)


def _move(states: frozenset[int], sym: str, transitions: dict[int, dict[str | None, set[int]]]) -> set[int]:
    out: set[int] = set()
    for s in states:
        out.update(transitions.get(s, {}).get(sym, set()))
    return out


def make_regex_dfa(regex: str) -> MinimalDFA:
    """Build a minimized DFA from a user-provided regular expression.

    The returned object is compatible with the rest of this project exactly like
    the random DFA returned by make_random_dfa.
    """
    start, accept, nfa_transitions, alphabet = _regex_postfix_to_nfa(_regex_to_postfix(regex))
    if not alphabet:
        # The project expects a non-empty alphabet. For pure ε expressions, use
        # a default binary alphabet so the LLM can still ask membership queries.
        alphabet = {"a", "b"}

    alphabet_sorted = sorted(alphabet)
    dfa_state_map: dict[frozenset[int], int] = {}
    dfa_transitions: dict[int, dict[str, int]] = {}
    dfa_final_states: set[int] = set()

    start_set = _epsilon_closure({start}, nfa_transitions)
    dfa_state_map[start_set] = 0
    queue: list[frozenset[int]] = [start_set]

    while queue:
        cur = queue.pop(0)
        cur_id = dfa_state_map[cur]
        dfa_transitions.setdefault(cur_id, {})

        if accept in cur:
            dfa_final_states.add(cur_id)

        for sym in alphabet_sorted:
            nxt = _epsilon_closure(_move(cur, sym, nfa_transitions), nfa_transitions)
            if nxt not in dfa_state_map:
                dfa_state_map[nxt] = len(dfa_state_map)
                queue.append(nxt)
            dfa_transitions[cur_id][sym] = dfa_state_map[nxt]

    states = set(dfa_state_map.values())
    return MinimalDFA.from_params(
        states=states,
        input_symbols=set(alphabet_sorted),
        transitions=dfa_transitions,
        initial_state=0,
        final_states=dfa_final_states,
        allow_partial=False,
        minimize=True,
        make_total=True,
    )


def bounds_exact(n, k):
    LB = pow(2, n-2) * pow(n, (k-1)*n)
    UB = pow(2, n) * pow(n, k*n) // math.factorial(n-1)
    return LB, UB


def _alphabet(k: int) -> List[str]:
    base = [chr(ord("a") + i) for i in range(26)]
    if k <= 26:
        return base[:k]
    out = base[:]
    i = 0
    while len(out) < k:
        out.append(f"a{i}")
        i += 1
    return out[:k]


def _zeta_k(k: int) -> float:
    if k <= 1:
        return 1e-9
    z = float(k) - 1e-6
    for _ in range(60):
        ez = math.exp(z)
        f = (z - k) * ez + k
        fp = (z - k + 1.0) * ez
        if fp == 0:
            break
        z2 = z - f / fp
        if abs(z2 - z) < 1e-14:
            break
        z = z2
    return z


def _nonzero_poisson(rng: random.Random, x: float) -> int:
    exm1 = math.expm1(x)
    k = 1
    p = x / exm1
    u = rng.random()
    while u >= p:
        u -= p
        k += 1
        p = x * p / k
    return k


def _sample_partition_sizes_exact_sum(rng: random.Random, n: int, k: int, x: float) -> List[int]:
    target = n * k
    while True:
        sizes = [_nonzero_poisson(rng, x) for _ in range(n)]
        if sum(sizes) == target:
            return sizes


def _sample_uniform_partition(rng: random.Random, total: int, sizes: List[int]) -> List[Set[int]]:
    labels = list(range(1, total + 1))
    rng.shuffle(labels)
    blocks: List[Set[int]] = []
    idx = 0
    for s in sizes:
        blocks.append(set(labels[idx:idx + s]))
        idx += s
    return blocks


def _partition_to_part_array(blocks: List[Set[int]], total: int) -> List[int]:
    order = sorted(range(len(blocks)), key=lambda i: min(blocks[i]))
    part = [0] * (total + 1)
    for new_id, i in enumerate(order, start=1):
        for x in blocks[i]:
            part[x] = new_id
    return part


def _part_array_to_boxed_diagram(part: List[int], n: int, m: int) -> Tuple[List[int], List[int]]:
    total = n + m
    mx = [0] * (total + 1)
    cur = 0
    for i in range(1, total + 1):
        cur = max(cur, part[i])
        mx[i] = cur

    first = {}
    for i in range(1, total + 1):
        if mx[i] not in first:
            first[mx[i]] = i

    remove = set(first.values())
    Max, Boxed = [], []
    for i in range(1, total + 1):
        if i not in remove:
            Max.append(mx[i])
            Boxed.append(part[i])
    return Max, Boxed


def _is_k_dyck(Max: List[int], k: int) -> bool:
    if k <= 1:
        return True
    km1 = k - 1
    for i, x in enumerate(Max, start=1):
        if x < (i + km1 - 1) // km1:
            return False
    return True


def _kdick_boxed_to_transition_structure(
    Max: List[int],
    Boxed: List[int],
    sigma: List[str],
) -> Tuple[Set[int], Dict[int, Dict[str, int]]]:

    stack: List[Tuple[int, str]] = []
    q_last = 1
    states = {1}
    transitions: Dict[int, Dict[str, int]] = {1: {}}

    for a in reversed(sigma):
        stack.append((1, a))

    i = 0
    j = 1

    while stack:
        p, a = stack.pop()
        transitions.setdefault(p, {})
        if j < Max[i]:
            q_last += 1
            states.add(q_last)
            transitions[q_last] = {}
            transitions[p][a] = q_last
            for a2 in reversed(sigma):
                stack.append((q_last, a2))
            j += 1
        else:
            transitions[p][a] = Boxed[i]
            i += 1

    return states, transitions


def boltzmann_sample_accessible_dfa(n: int, k: int, seed: int | None = None) -> MinimalDFA:
    rng = random.Random(seed)
    sigma = _alphabet(k)
    z = _zeta_k(k)

    m = (k - 1) * n
    total = k * n

    while True:
        sizes = _sample_partition_sizes_exact_sum(rng, n, k, z)
        blocks = _sample_uniform_partition(rng, total, sizes)
        part = _partition_to_part_array(blocks, total)
        Max, Boxed = _part_array_to_boxed_diagram(part, n, m)

        if len(Max) != m or not _is_k_dyck(Max, k):
            continue

        Max.append(n)
        Boxed.append(rng.randint(1, n))

        states, transitions = _kdick_boxed_to_transition_structure(Max, Boxed, sigma)
        if len(states) != n:
            continue

        finals = {q for q in states if rng.getrandbits(1)}

        dfa = MinimalDFA.from_params(
            states=states,
            input_symbols=set(sigma),
            transitions=transitions,
            initial_state=1,
            final_states=finals,
            allow_partial=False,
            minimize=True,
            make_total=False,
        )

        if len(dfa.states) != n:
            continue

        return dfa