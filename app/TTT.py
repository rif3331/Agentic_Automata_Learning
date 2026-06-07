"""
Responsible for:

- Implementing the TTT learning algorithm for DFA inference
- Handling membership and equivalence queries through teacher/oracle components
- Managing discrimination trees, hypothesis refinement, and counterexample processing
- Tracking query history, caching repeated membership queries, and counting query costs
- Running the TTT strategy and producing DFA learning performance results
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from collections import OrderedDict

from automata.fa.dfa import DFA
from aalpy.base import SUL


from html_code.draw_DFA_html import draw_DFA_html
from html_code.ttt_comparison_html import write_ttt_comparison_html


from utils import (
    _word_to_seq,
    _word_repr,
)


Word = Tuple[Any, ...]

TTT_RUNTIME_NAME = "TTT"


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


class _LRU:
    def __init__(self, max_size: Optional[int]):
        self.max_size = max_size
        self._d: "OrderedDict[Word, bool]" = OrderedDict()

    def get(self, k: Word) -> Optional[bool]:
        if k not in self._d:
            return None
        v = self._d.pop(k)
        self._d[k] = v
        return v

    def put(self, k: Word, v: bool):
        if k in self._d:
            self._d.pop(k)
        self._d[k] = v
        if self.max_size is not None and self.max_size > 0:
            while len(self._d) > self.max_size:
                self._d.popitem(last=False)


class CountingSUL(SUL):
    def __init__(
        self,
        sul: DFATeacherSUL,
        history: Optional[List[Any]] = None,
        cache_max_size: Optional[int] = None,
        knowledge_state: Optional[dict[str, Any]] = None,
        dfa: Optional[DFA] = None,
        strategy_name: str = TTT_RUNTIME_NAME,
    ):
        super().__init__()
        self.sul = sul
        self.mq_queries = 0
        self.history = history
        self._lru = _LRU(cache_max_size)
        self.knowledge_state = knowledge_state if knowledge_state is not None else {}
        self.dfa = dfa
        self.strategy_name = strategy_name

    def pre(self):
        return self.sul.pre()

    def post(self):
        return self.sul.post()

    def step(self, inp):
        return self.sul.step(inp)

    def membership(self, word: Sequence[Any]) -> bool:
        key = tuple(word) if word is not None else tuple()

        cached = self._lru.get(key)
        if cached is not None:
            return bool(cached)

        w = _word_repr(list(key))

        mem = bool(self.sul.accepts_word(key))

        self.mq_queries += 1
        self._lru.put(key, mem)

        if self.history is not None:
            self.history.append(QueryLogItem("MQ", w, mem, None))

        return mem

    def query(self, word: Sequence[Any]):
        mem = self.membership(word)
        return [mem]


class MinimalDFAEqOracle:
    def __init__(
        self,
        target_dfa: DFA,
        alphabet: Optional[List[Any]] = None,
        history: Optional[List[Any]] = None,
        knowledge_state: Optional[dict[str, Any]] = None,
        strategy_name: str = TTT_RUNTIME_NAME,
        minimal_counterexample: bool = False,
        counterexample_max_extra_len: int = 3,
    ):
        from dfa_class import MinimalDFA

        self.target_dfa = target_dfa
        self.minimal_counterexample = bool(minimal_counterexample)
        self.counterexample_max_extra_len = int(counterexample_max_extra_len)
        raw_counterexample_mode = int(getattr(target_dfa, "counterexample_mode", 0))
        self.counterexample_mode = 0 if raw_counterexample_mode == 2 else raw_counterexample_mode

        self.knowledge_state = knowledge_state if knowledge_state is not None else {}
        self.strategy_name = strategy_name

        self.target_min = MinimalDFA.from_dfa(
            target_dfa,
            run_strategy=False,
            minimal_counterexample=self.minimal_counterexample,
            counterexample_max_extra_len=self.counterexample_max_extra_len,
            counterexample_mode=self.counterexample_mode,
        )

        self.num_queries = 0
        self.history = history
        self.alphabet = sorted(alphabet, key=str) if alphabet is not None else None

    def reset(self):
        return

    def find_cex(self, hypothesis: DFA):
        from dfa_class import MinimalDFA

        self.num_queries += 1

        hyp_min = MinimalDFA.from_dfa(
            hypothesis,
            run_strategy=False,
            minimal_counterexample=self.minimal_counterexample,
            counterexample_max_extra_len=self.counterexample_max_extra_len,
            counterexample_mode=self.counterexample_mode,
        )
        ok, w = self.target_min.eq(hyp_min)

        seq = _word_to_seq(w, self.alphabet)
        cex = _word_repr(seq) if not ok else None

        link = write_ttt_comparison_html(
            self.target_dfa,
            hypothesis,
            ok=bool(ok),
            counterexample=cex,
        )

        if ok:
            if self.history is not None:
                self.history.append(QueryLogItem("EQ", None, True, link, hypothesis))
            return None

        if self.history is not None:
            self.history.append(QueryLogItem("EQ", cex, False, link, hypothesis))

        return tuple(seq)


class _DTNode:
    __slots__ = ("disc", "left", "right", "state")

    def __init__(
        self,
        disc: Optional[Word] = None,
        left: Optional["_DTNode"] = None,
        right: Optional["_DTNode"] = None,
        state: Optional[int] = None,
    ):
        self.disc = disc
        self.left = left
        self.right = right
        self.state = state

    def is_leaf(self) -> bool:
        return self.state is not None


class _DiscriminationTree:
    def __init__(self, mq: Callable[[Word], bool]):
        self.mq = mq
        self.root = _DTNode(state=0)

    def sift(self, u: Word) -> int:
        node = self.root
        while not node.is_leaf():
            v = node.disc or tuple()
            out = bool(self.mq(u + v))
            node = node.right if out else node.left
        return int(node.state)

    def _replace_leaf(self, node: _DTNode, q: int, new_node: _DTNode) -> bool:
        if node.is_leaf():
            return False
        if node.left is not None and node.left.is_leaf() and int(node.left.state) == q:
            node.left = new_node
            return True
        if node.right is not None and node.right.is_leaf() and int(node.right.state) == q:
            node.right = new_node
            return True
        if node.left is not None:
            if self._replace_leaf(node.left, q, new_node):
                return True
        if node.right is not None:
            if self._replace_leaf(node.right, q, new_node):
                return True
        return False

    def _replace_leaf_with_undo(self, node: _DTNode, q: int, new_node: _DTNode) -> Optional[Tuple[_DTNode, str, _DTNode]]:
        if node.is_leaf():
            return None
        if node.left is not None and node.left.is_leaf() and int(node.left.state) == q:
            old = node.left
            node.left = new_node
            return (node, "left", old)
        if node.right is not None and node.right.is_leaf() and int(node.right.state) == q:
            old = node.right
            node.right = new_node
            return (node, "right", old)
        if node.left is not None:
            r = self._replace_leaf_with_undo(node.left, q, new_node)
            if r is not None:
                return r
        if node.right is not None:
            r = self._replace_leaf_with_undo(node.right, q, new_node)
            if r is not None:
                return r
        return None

    def split_leaf(self, qold: int, qnew: int, disc: Word, access_old: Word, access_new: Word):
        out_old = bool(self.mq(access_old + disc))
        out_new = bool(self.mq(access_new + disc))
        if out_old == out_new:
            raise RuntimeError("split_leaf: discriminator does not separate qold/qnew")

        internal = _DTNode(disc=disc)
        if out_old:
            internal.right = _DTNode(state=qold)
            internal.left = _DTNode(state=qnew)
        else:
            internal.left = _DTNode(state=qold)
            internal.right = _DTNode(state=qnew)

        if self.root.is_leaf() and int(self.root.state) == qold:
            self.root = internal
            return

        if not self._replace_leaf(self.root, qold, internal):
            raise RuntimeError("split_leaf failed: leaf not found")

    def minimize_suffix(self, u1: Word, u2: Word, disc: Word) -> Word:
        if not disc:
            return disc
        for i in range(len(disc)):
            cand = disc[i:]
            if bool(self.mq(u1 + cand)) != bool(self.mq(u2 + cand)):
                return cand
        return disc

    def path_signature(self, u: Word) -> List[Tuple[Word, bool]]:
        sig: List[Tuple[Word, bool]] = []
        node = self.root
        while not node.is_leaf():
            v = node.disc or tuple()
            out = bool(self.mq(u + v))
            sig.append((v, out))
            node = node.right if out else node.left
        return sig

    def all_discriminators(self) -> List[Word]:
        out: List[Word] = []
        stack = [self.root]
        seen = set()
        while stack:
            n = stack.pop()
            if n is None:
                continue
            if not n.is_leaf():
                d = n.disc or tuple()
                if d not in seen:
                    seen.add(d)
                    out.append(d)
                stack.append(n.left)
                stack.append(n.right)
        return out


class _Hypothesis:
    def __init__(self, alphabet: List[Any], n_states: int):
        self.alphabet = list(alphabet)
        self.n_states = n_states
        self.delta: List[Dict[Any, int]] = [dict() for _ in range(n_states)]
        self.acc: List[bool] = [False] * n_states

    def run(self, w: Word) -> int:
        q = 0
        for a in w:
            q = self.delta[q][a]
        return q

    def to_dfa(self) -> DFA:
        states = {f"q{i}" for i in range(self.n_states)}
        transitions: Dict[str, Dict[Any, str]] = {}
        finals = set()

        for i in range(self.n_states):
            qi = f"q{i}"
            transitions[qi] = {a: f"q{self.delta[i][a]}" for a in self.alphabet}
            if self.acc[i]:
                finals.add(qi)

        return DFA(
            states=states,
            input_symbols=set(self.alphabet),
            transitions=transitions,
            initial_state="q0",
            final_states=finals,
            allow_partial=True,
        )

    def accepts(self, w: Word) -> bool:
        q = self.run(w)
        return bool(self.acc[q])


class TTTLearner:
    def __init__(self, alphabet: List[Any], sul: CountingSUL, eqo: MinimalDFAEqOracle):
        self.alphabet = list(alphabet)
        self.sul = sul
        self.eqo = eqo
        self.access: Dict[int, Word] = {0: tuple()}
        self.n_states = 1
        self.dt = _DiscriminationTree(self._mq)

    def _mq(self, w: Word) -> bool:
        return bool(self.sul.membership(list(w)))

    def _build_hypothesis(self) -> _Hypothesis:
        H = _Hypothesis(self.alphabet, self.n_states)

        for q in range(self.n_states):
            H.acc[q] = bool(self._mq(self.access[q]))

        for q in range(self.n_states):
            u = self.access[q]
            for a in self.alphabet:
                H.delta[q][a] = self.dt.sift(u + (a,))

        return H

    def _rs_decompose(self, H: _Hypothesis, w: Word) -> Tuple[Word, Any, Word]:
        m = len(w)
        if m == 0:
            return (tuple(), None, tuple())

        q_after: List[int] = [0]
        for i in range(m):
            q_after.append(H.run(w[: i + 1]))

        def g(i: int) -> bool:
            q = q_after[i]
            return bool(self._mq(self.access[q] + w[i:]))

        outs = [g(i) for i in range(m + 1)]

        idx = None
        for i in range(m):
            if outs[i] != outs[i + 1]:
                idx = i
                break

        if idx is None:
            idx = m - 1

        u = w[:idx]
        a = w[idx]
        v = w[idx + 1:]
        return (u, a, v)

    def _candidate_discriminators(self, a: Any, v: Word) -> List[Word]:
        cands: List[Word] = []
        vv = tuple(v)

        for i in range(len(vv) + 1):
            cands.append(vv[i:])

        av = (a,) + vv

        for i in range(len(av) + 1):
            cands.append(av[i:])

        cands.append((a,) + vv)

        for i in range(len(vv) + 1):
            cands.append((a,) + vv[i:])

        cands.append((a,))
        cands.append(tuple())

        for d in self.dt.all_discriminators():
            cands.append(tuple(d))

        uniq: List[Word] = []
        seen = set()

        for x in cands:
            if x not in seen:
                seen.add(x)
                uniq.append(x)

        return uniq

    def _pick_progress_discriminator(self, q_u: int, a: Any, qold: int, qnew: int, v: Word) -> Word:
        before_sig = self.dt.path_signature(self.access[q_u] + (a,))
        before_leaf = self.dt.sift(self.access[q_u] + (a,))

        best_sep: Optional[Word] = None

        for cand in self._candidate_discriminators(a, v):
            if bool(self._mq(self.access[qold] + cand)) == bool(self._mq(self.access[qnew] + cand)):
                continue

            cand2 = self.dt.minimize_suffix(self.access[qold], self.access[qnew], cand)

            if bool(self._mq(self.access[qold] + cand2)) == bool(self._mq(self.access[qnew] + cand2)):
                continue

            if best_sep is None:
                best_sep = cand2

            internal = _DTNode(disc=cand2)
            out_old = bool(self._mq(self.access[qold] + cand2))

            if out_old:
                internal.right = _DTNode(state=qold)
                internal.left = _DTNode(state=qnew)
            else:
                internal.left = _DTNode(state=qold)
                internal.right = _DTNode(state=qnew)

            if self.dt.root.is_leaf() and int(self.dt.root.state) == qold:
                old_root = self.dt.root
                self.dt.root = internal

                after_leaf = self.dt.sift(self.access[q_u] + (a,))
                after_sig = self.dt.path_signature(self.access[q_u] + (a,))

                self.dt.root = old_root

                if after_leaf != before_leaf or after_sig != before_sig:
                    return cand2

                continue

            undo = self.dt._replace_leaf_with_undo(self.dt.root, qold, internal)

            if undo is None:
                continue

            after_leaf = self.dt.sift(self.access[q_u] + (a,))
            after_sig = self.dt.path_signature(self.access[q_u] + (a,))

            parent, side, old_child = undo

            if side == "left":
                parent.left = old_child
            else:
                parent.right = old_child

            if after_leaf != before_leaf or after_sig != before_sig:
                return cand2

        if best_sep is not None:
            return best_sep

        raise RuntimeError("failed to find progress discriminator")

    def _refine(self, H: _Hypothesis, w: Word):
        u, a, v = self._rs_decompose(H, w)

        if a is None:
            raise RuntimeError("empty counterexample")

        q_u = H.run(u)
        qold = H.run(u + (a,))

        qnew = self.n_states
        self.n_states += 1
        self.access[qnew] = self.access[q_u] + (a,)

        disc = self._pick_progress_discriminator(
            q_u=q_u,
            a=a,
            qold=qold,
            qnew=qnew,
            v=tuple(v),
        )

        disc = self.dt.minimize_suffix(
            self.access[qold],
            self.access[qnew],
            disc,
        )

        self.dt.split_leaf(
            qold=qold,
            qnew=qnew,
            disc=disc,
            access_old=self.access[qold],
            access_new=self.access[qnew],
        )

    def _target_accepts(self, w: Word) -> bool:
        return bool(self._mq(w))

    def run(self, max_rounds: int = 10000, max_internal_refine: int = 1000) -> DFA:
        rounds = 0

        while True:
            rounds += 1

            if rounds > max_rounds:
                raise RuntimeError("TTT exceeded max_rounds")

            H = self._build_hypothesis()
            hyp_dfa = H.to_dfa()
            cex = self.eqo.find_cex(hyp_dfa)

            if cex is None:
                return hyp_dfa

            cex_t = tuple(cex)

            for _ in range(max_internal_refine):
                H = self._build_hypothesis()
                hyp_dfa = H.to_dfa()

                hyp_accept = bool(hyp_dfa.accepts_input(list(cex_t)))
                tgt_accept = bool(self._target_accepts(cex_t))

                if hyp_accept == tgt_accept:
                    break

                self._refine(H, cex_t)
            else:
                raise RuntimeError("TTT stuck: counterexample not eliminated by internal refinement")


class TTTStrategy:
    def __init__(self, print_level: int = 0, cache_max_size: Optional[int] = None):
        self.print_level = print_level
        self.cache_max_size = cache_max_size

    def run(self, dfa):

        from game_types import DFAStrategy

        history: List[Any] = []

        alphabet = sorted(dfa.input_symbols, key=str)

        teacher = DFATeacherSUL(dfa)

        sul = CountingSUL(
            teacher,
            history=history,
            cache_max_size=self.cache_max_size,
        )

        eqo = MinimalDFAEqOracle(
            dfa,
            alphabet=alphabet,
            history=history,
            minimal_counterexample=bool(getattr(dfa, "minimal_counterexample", False)),
            counterexample_max_extra_len=int(getattr(dfa, "counterexample_max_extra_len", 3)),
        )

        learner = TTTLearner(alphabet, sul, eqo)
        learner.run()

        eq_q = eqo.num_queries
        mq_q = sul.mq_queries
        total = eq_q + mq_q

        return DFAStrategy.Result(
            total_queries=total,
            eq_queries=eq_q,
            mq_queries=mq_q,
            history=history,
        )