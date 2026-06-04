"""
Responsible for:

- Implementing the interactive tools available to the LLM during DFA learning
- Handling membership queries for checking whether words belong to the target language
- Evaluating candidate DFAs submitted by the model against the target DFA
- Validating tool inputs, updating knowledge state, and managing counterexamples
- Generating DFA comparison reports and equivalence evaluation outputs
"""
from game_types import ToolInterface, ToolInput, ToolOutput, EvaluationToolInterface
from constants import TOOL_PROMPTS
from dfa_class import MinimalDFA
from utils import (
    _tokenize_by_vocab,
    _normalize_word_display,
)
from html_code.llm_comparison_html import (
    write_llm_comparison_html,
    read_if_path,
)


def print_llm_current_word_dictionaries(ks: dict) -> None:
    accepted = sorted(str(w) for w in ks.get("words_accepted_by_dfa", set()))
    rejected = sorted(str(w) for w in ks.get("words_rejected_by_dfa", set()))
    counts = ks.get("equivalence_class_counts", {})

    # print("\nLLM CURRENT WORD DICTIONARIES")
    # print("ACCEPTED WORDS DICT:", accepted)
    # print("REJECTED WORDS DICT:", rejected)
    # print("EQUIVALENCE CLASS COUNTS:", counts)
    # print("END LLM CURRENT WORD DICTIONARIES", flush=True)


class IsWordInLanguageTool(ToolInterface):
    tool_name = "is_word_in_language"
    prompt = TOOL_PROMPTS[tool_name]

    def invoke(self, request: ToolInput) -> ToolOutput:
        ks = request.get("knowledge_state") or {
            "words_accepted_by_dfa": set(),
            "words_rejected_by_dfa": set(),
        }
        # print_equivalence_class_counts_from_tool_reply({"knowledge_state": ks})
        # print_counterexamples_so_far()
        # print_counterexample_statistics()
        # print_llm_current_word_dictionaries(ks)

        if request.get("tool_name") != self.tool_name:
            return {
                "tool_name": self.tool_name,
                "call_count": request.get("call_count", 0),
                "error": "TOOL_NAME_MISMATCH",
                "output": None,
                "knowledge_state": ks,
            }

        call_count = request["call_count"]
        payload = request.get("input") or request.get("parameters") or {}
        word = payload.get("word")

        if not isinstance(word, str):
            return {
                "tool_name": self.tool_name,
                "call_count": call_count,
                "error": "INVALID_INPUT: 'word' must be a string",
                "output": None,
                "knowledge_state": ks,
            }

        game = getattr(self, "game", None)
        if game is None:
            return {
                "tool_name": self.tool_name,
                "call_count": call_count,
                "error": "GAME_NOT_ATTACHED",
                "output": None,
                "knowledge_state": ks,
            }

        if game.dfa is None:
            return {
                "tool_name": self.tool_name,
                "call_count": call_count,
                "error": "DFA_NOT_READY",
                "output": None,
                "knowledge_state": ks,
            }

        display_word = "ε" if word == "" else word

        alphabet = getattr(game.dfa, "input_symbols", None)
        alphabet_set = None
        if isinstance(alphabet, set):
            alphabet_set = alphabet
        else:
            try:
                alphabet_set = set(alphabet)
            except Exception:
                alphabet_set = None

        tokens, bad_chunk = _tokenize_by_vocab(word, alphabet_set if isinstance(alphabet_set, set) else None)
        if tokens is None:
            bad = "" if bad_chunk is None else bad_chunk
            return {
                "tool_name": self.tool_name,
                "call_count": call_count,
                "error": f"SYMBOL_NOT_IN_ALPHABET: '{bad}' is not in DFA alphabet",
                "output": {"word": display_word, "invalid_symbol": bad},
                "knowledge_state": ks,
            }

        if alphabet_set is not None:
            for sym in tokens:
                if sym not in alphabet_set:
                    return {
                        "tool_name": self.tool_name,
                        "call_count": call_count,
                        "error": f"SYMBOL_NOT_IN_ALPHABET: '{sym}' is not in DFA alphabet",
                        "output": {"word": display_word, "invalid_symbol": sym},
                        "knowledge_state": ks,
                    }

        accepted = bool(game.dfa.accepts_input(tokens))
        if accepted:
            ks["words_accepted_by_dfa"].add(display_word)
        else:
            ks["words_rejected_by_dfa"].add(display_word)

        # print_llm_current_word_dictionaries(ks)

        return {
            "tool_name": self.tool_name,
            "call_count": call_count,
            "error": None,
            "output": {"word": display_word, "accepted": accepted},
            "knowledge_state": ks,
        }


class EvaluateDFACandidateTool(EvaluationToolInterface):
    tool_name = "evaluate_dfa_candidate"
    prompt = TOOL_PROMPTS[tool_name]

    def __init__(self):
        self._last_report_path = ""

    def draw(self) -> str:
        return self._last_report_path

    def invoke(self, request: ToolInput) -> ToolOutput:
        ks = request.get("knowledge_state") or {
            "words_accepted_by_dfa": set(),
            "words_rejected_by_dfa": set(),
        }
        # print_equivalence_class_counts_from_tool_reply({"knowledge_state": ks})
        # print_counterexamples_so_far()
        # print_counterexample_statistics()
        # print_llm_current_word_dictionaries(ks)

        if request.get("tool_name") != self.tool_name:
            return {
                "tool_name": self.tool_name,
                "call_count": request.get("call_count", 0),
                "error": "TOOL_NAME_MISMATCH",
                "output": None,
                "knowledge_state": ks,
            }

        call_count = request["call_count"]
        payload = request.get("input") or request.get("parameters") or {}
        cand = payload.get("candidate_dfa")

        if not isinstance(cand, dict):
            return {
                "tool_name": self.tool_name,
                "call_count": call_count,
                "error": "INVALID_INPUT: 'candidate_dfa' must be an object",
                "output": None,
                "knowledge_state": ks,
            }

        states = cand.get("states")
        alphabet = cand.get("alphabet")
        start_state = cand.get("start_state")
        accept_states = cand.get("accept_states")
        transitions_list = cand.get("transitions")

        if not isinstance(states, list) or not all(isinstance(x, str) for x in states):
            return {
                "tool_name": self.tool_name,
                "call_count": call_count,
                "error": "INVALID_INPUT: 'states' must be a list[str]",
                "output": None,
                "knowledge_state": ks,
            }

        if not isinstance(alphabet, list) or not all(isinstance(x, str) for x in alphabet):
            return {
                "tool_name": self.tool_name,
                "call_count": call_count,
                "error": "INVALID_INPUT: 'alphabet' must be a list[str]",
                "output": None,
                "knowledge_state": ks,
            }

        if not isinstance(start_state, str):
            return {
                "tool_name": self.tool_name,
                "call_count": call_count,
                "error": "INVALID_INPUT: 'start_state' must be a string",
                "output": None,
                "knowledge_state": ks,
            }

        if not isinstance(accept_states, list) or not all(isinstance(x, str) for x in accept_states):
            return {
                "tool_name": self.tool_name,
                "call_count": call_count,
                "error": "INVALID_INPUT: 'accept_states' must be a list[str]",
                "output": None,
                "knowledge_state": ks,
            }

        if not isinstance(transitions_list, list) or not all(
            isinstance(t, list) and len(t) == 3 and all(isinstance(v, str) for v in t) for t in transitions_list
        ):
            return {
                "tool_name": self.tool_name,
                "call_count": call_count,
                "error": "INVALID_INPUT: 'transitions' must be a list of [source_state, symbol, target_state]",
                "output": None,
                "knowledge_state": ks,
            }

        if start_state not in set(states):
            return {
                "tool_name": self.tool_name,
                "call_count": call_count,
                "error": "INVALID_INPUT: 'start_state' must be in 'states'",
                "output": None,
                "knowledge_state": ks,
            }

        states_set = set(states)
        alphabet_set = set(alphabet)
        accept_set = set(accept_states)

        if not accept_set.issubset(states_set):
            bad = next(iter(accept_set - states_set))
            return {
                "tool_name": self.tool_name,
                "call_count": call_count,
                "error": f"INVALID_INPUT: accept state '{bad}' is not in 'states'",
                "output": None,
                "knowledge_state": ks,
            }

        trans_dict = {s: {} for s in states_set}
        for src, sym, dst in transitions_list:
            if src not in states_set:
                return {
                    "tool_name": self.tool_name,
                    "call_count": call_count,
                    "error": f"INVALID_INPUT: transition source '{src}' is not in 'states'",
                    "output": None,
                    "knowledge_state": ks,
                }
            if dst not in states_set:
                return {
                    "tool_name": self.tool_name,
                    "call_count": call_count,
                    "error": f"INVALID_INPUT: transition target '{dst}' is not in 'states'",
                    "output": None,
                    "knowledge_state": ks,
                }
            if sym not in alphabet_set:
                return {
                    "tool_name": self.tool_name,
                    "call_count": call_count,
                    "error": f"INVALID_INPUT: transition symbol '{sym}' is not in 'alphabet'",
                    "output": None,
                    "knowledge_state": ks,
                }
            if sym in trans_dict[src]:
                return {
                    "tool_name": self.tool_name,
                    "call_count": call_count,
                    "error": f"INVALID_INPUT: nondeterministic transition from '{src}' on '{sym}'",
                    "output": None,
                    "knowledge_state": ks,
                }
            trans_dict[src][sym] = dst

        game = getattr(self, "game", None)
        if game is None:
            return {
                "tool_name": self.tool_name,
                "call_count": call_count,
                "error": "GAME_NOT_ATTACHED",
                "output": None,
                "knowledge_state": ks,
            }

        if game.dfa is None:
            return {
                "tool_name": self.tool_name,
                "call_count": call_count,
                "error": "DFA_NOT_READY",
                "output": None,
                "knowledge_state": ks,
            }

        try:
            candidate = MinimalDFA.from_params(
                states=states_set,
                input_symbols=alphabet_set,
                transitions=trans_dict,
                initial_state=start_state,
                final_states=accept_set,
                allow_partial=True,
                minimize=True,
                retain_names=False,
                make_total=True,
            )
        except Exception as e:
            return {
                "tool_name": self.tool_name,
                "call_count": call_count,
                "error": f"INVALID_CANDIDATE_DFA: {e}",
                "output": None,
                "knowledge_state": ks,
            }

        eq, witness = game.dfa.eq(candidate)

        witness_display = _normalize_word_display(witness)

        left_draw = game.dfa.draw()
        left_html = read_if_path(left_draw)

        right_draw = candidate.draw()

        from pathlib import Path
        import shutil

        _p = Path(right_draw)
        stable_hypothesis_path = _p.parent / f"{_p.stem}_{call_count}{_p.suffix}"

        shutil.copy2(str(_p), str(stable_hypothesis_path))

        print(
            f"HYPOTHESIS_DFA_LINK::CALL={call_count}::PATH={stable_hypothesis_path}",
            flush=True,
        )

        right_html = read_if_path(str(stable_hypothesis_path))

        report_path = write_llm_comparison_html(
            eq=bool(eq),
            witness_word=witness_display,
            left_html=left_html,
            right_html=right_html,
            call_count=call_count,
        )

        self._last_report_path = report_path

        if not eq:
            alpha = getattr(game.dfa, "input_symbols", None)
            alpha_set = None
            if isinstance(alpha, set):
                alpha_set = alpha
            else:
                try:
                    alpha_set = set(alpha)
                except Exception:
                    alpha_set = None

            if isinstance(witness, str):
                toks, _bad = _tokenize_by_vocab(witness, alpha_set if isinstance(alpha_set, set) else None)
                witness_seq = toks if toks is not None else [witness]
            elif isinstance(witness, (list, tuple)):
                witness_seq = list(witness)
            else:
                witness_seq = _tokenize_by_vocab(str(witness), alpha_set if isinstance(alpha_set, set) else None)[0] or []

            accepted = bool(game.dfa.accepts_input(witness_seq))
            if accepted:
                ks["words_accepted_by_dfa"].add(witness_display)
            else:
                ks["words_rejected_by_dfa"].add(witness_display)

        # print_llm_current_word_dictionaries(ks)

        out_payload = {
            "score": 1.0 if eq else 0.0,
            "optimal": bool(eq),
            "witness_word": "" if eq else witness_display,
            "html": report_path,
            "_candidate_obj": candidate,
        }

        return {
            "tool_name": self.tool_name,
            "call_count": call_count,
            "error": None,
            "output": out_payload,
            "knowledge_state": ks,
        }