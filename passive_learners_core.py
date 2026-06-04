"""
Standalone passive learner helpers for runtime passive/gold analysis.

This file intentionally does not import add_knowledge_mode_general_en.py.
It contains only the small set of utilities needed to run RPNI / EDSM /
Blue-Fringe on membership observations and compare the learned DFA to the
runtime target DFA.
"""
from __future__ import annotations

import html
import re
from typing import Any, Dict, Optional, Tuple

try:
    import aalpy.learning_algs as _aalpy_learning_algs
    run_RPNI = getattr(_aalpy_learning_algs, "run_RPNI", None)
    run_EDSM = getattr(_aalpy_learning_algs, "run_EDSM", None)
    run_BlueFringe = (
        getattr(_aalpy_learning_algs, "run_BlueFringe", None)
        or getattr(_aalpy_learning_algs, "run_Blue_Fringe", None)
        or getattr(_aalpy_learning_algs, "run_BLUE_FRINGE", None)
        or getattr(_aalpy_learning_algs, "run_blue_fringe", None)
    )
    try:
        from aalpy.learning_algs.general_passive.GeneralizedStateMerging import run_GSM as _run_GSM
        from aalpy.utils.HelperFunctions import dfa_from_moore as _dfa_from_moore
    except Exception:
        _run_GSM = None
        _dfa_from_moore = None
except Exception:
    run_RPNI = None
    run_EDSM = None
    run_BlueFringe = None
    _run_GSM = None
    _dfa_from_moore = None


def normalize_word(raw: Any) -> str:
    """Normalize words exactly for passive learner observations."""
    if raw is None:
        return ""
    s = html.unescape(str(raw)).strip()
    if s in {"", "ε", "epsilon", "eps", "<eps>"}:
        return ""
    s = re.sub(r"\s+", "", s)
    return s


def word_to_tuple_for_passive(word: Any) -> Tuple[str, ...]:
    if word is None:
        return tuple()
    if isinstance(word, tuple):
        return tuple(str(x) for x in word)
    if isinstance(word, list):
        return tuple(str(x) for x in word)
    word = normalize_word(word)
    if word in {"", "ε"}:
        return tuple()
    return tuple(word)


def state_name_for_passive(state: Any) -> str:
    return str(getattr(state, "state_id", getattr(state, "name", state)))


def aalpy_state_is_accepting(state: Any) -> bool:
    for attr in ["is_accepting", "accepting"]:
        if hasattr(state, attr):
            value = getattr(state, attr)
            if callable(value):
                try:
                    return bool(value())
                except Exception:
                    pass
            else:
                return bool(value)
    if hasattr(state, "output"):
        return bool(getattr(state, "output"))
    return False


def _run_aalpy_passive_learner(learner_fn, data):
    errors = []
    for kwargs in [
        {"automaton_type": "dfa", "input_completeness": "sink_state", "print_info": False},
        {"automaton_type": "dfa", "print_info": False},
        {"automaton_type": "dfa"},
        {"print_info": False},
        {},
    ]:
        try:
            return learner_fn(data, **kwargs)
        except TypeError as exc:
            errors.append(str(exc))
            continue
    raise TypeError("Could not call AALpy passive learner with supported signatures: " + " | ".join(errors))


def _run_blue_fringe_fallback(data, automaton_type="dfa", input_completeness="sink_state", print_info=False):
    if _run_GSM is None or _dfa_from_moore is None:
        raise RuntimeError(
            "AALpy Blue-Fringe fallback needs AALpy general_passive.run_GSM. "
            "Install/upgrade AALpy: pip install -U aalpy"
        )
    if automaton_type != "dfa":
        raise ValueError("Blue-Fringe fallback currently supports only DFA learning.")
    learned_model = _run_GSM(
        data,
        output_behavior="moore",
        transition_behavior="deterministic",
        data_format="labeled_sequences",
        instrumentation=None,
    )
    learned_model = _dfa_from_moore(learned_model)
    if hasattr(learned_model, "is_input_complete") and not learned_model.is_input_complete():
        if hasattr(learned_model, "make_input_complete") and input_completeness:
            learned_model.make_input_complete(input_completeness)
    return learned_model


def infer_minimal_dfa_with_aalpy_passive(
    template_dfa,
    accepted_words: Dict[Any, Any],
    rejected_words: Dict[Any, Any],
    *,
    algorithm_name: str,
    learner_fn,
):
    if learner_fn is None:
        raise RuntimeError(f"AALpy learner for {algorithm_name} is not available. Run: pip install aalpy")

    positive = {word_to_tuple_for_passive(w) for w in accepted_words.keys()}
    negative = {word_to_tuple_for_passive(w) for w in rejected_words.keys()}

    overlap = positive & negative
    if overlap:
        raise ValueError(f"Words appear in both accepted and rejected sets: {overlap}")
    if not positive and not negative:
        raise ValueError(f"{algorithm_name} needs at least one accepted or rejected word.")

    data = [(w, True) for w in positive] + [(w, False) for w in negative]
    learned = _run_aalpy_passive_learner(learner_fn, data)

    if hasattr(learned, "make_input_complete"):
        try:
            learned.make_input_complete("sink_state")
        except TypeError:
            learned.make_input_complete()

    states = set()
    final_states = set()
    transitions = {}
    input_symbols = set(getattr(template_dfa, "input_symbols", set()))

    if hasattr(learned, "get_input_alphabet"):
        input_symbols |= {str(x) for x in learned.get_input_alphabet()}

    for state in learned.states:
        s_name = state_name_for_passive(state)
        states.add(s_name)
        transitions.setdefault(s_name, {})

        if aalpy_state_is_accepting(state):
            final_states.add(s_name)

        for symbol, target in getattr(state, "transitions", {}).items():
            symbol = str(symbol)
            t_name = state_name_for_passive(target)
            input_symbols.add(symbol)
            states.add(t_name)
            transitions.setdefault(s_name, {})[symbol] = t_name
            transitions.setdefault(t_name, {})

    initial_state = state_name_for_passive(learned.initial_state)

    return type(template_dfa).from_params(
        states=states,
        input_symbols=input_symbols,
        transitions=transitions,
        initial_state=initial_state,
        final_states=final_states,
        allow_partial=True,
        minimize=True,
        retain_names=False,
        make_total=True,
        grid=getattr(template_dfa, "grid", False),
        check_locality=getattr(template_dfa, "check_locality", False),
        search_ngram_approx=getattr(template_dfa, "search_ngram_approx", False),
        minimal_counterexample=getattr(template_dfa, "minimal_counterexample", False),
        counterexample_max_extra_len=getattr(template_dfa, "counterexample_max_extra_len", 3),
    )


def infer_minimal_dfa_with_rpni(template_dfa, accepted_words: Dict[Any, Any], rejected_words: Dict[Any, Any]):
    return infer_minimal_dfa_with_aalpy_passive(
        template_dfa,
        accepted_words,
        rejected_words,
        algorithm_name="RPNI",
        learner_fn=run_RPNI,
    )


def infer_minimal_dfa_with_edsm(template_dfa, accepted_words: Dict[Any, Any], rejected_words: Dict[Any, Any]):
    return infer_minimal_dfa_with_aalpy_passive(
        template_dfa,
        accepted_words,
        rejected_words,
        algorithm_name="EDSM",
        learner_fn=run_EDSM,
    )


def infer_minimal_dfa_with_blue_fringe(template_dfa, accepted_words: Dict[Any, Any], rejected_words: Dict[Any, Any]):
    learner = run_BlueFringe if run_BlueFringe is not None else _run_blue_fringe_fallback
    return infer_minimal_dfa_with_aalpy_passive(
        template_dfa,
        accepted_words,
        rejected_words,
        algorithm_name="Blue-Fringe",
        learner_fn=learner,
    )


def are_dfas_equivalent(candidate_dfa, target_dfa) -> tuple[Optional[bool], str]:
    if candidate_dfa is None:
        return None, "Cannot compare: candidate DFA was not created."
    if target_dfa is None:
        return None, "Cannot compare: target DFA was not created."
    try:
        equivalent, witness = target_dfa.eq(candidate_dfa)
    except Exception:
        try:
            equivalent, witness = candidate_dfa.eq(target_dfa)
        except Exception as exc:
            return None, f"Cannot compare DFA objects: {exc}"
    if equivalent:
        return True, "Equivalence check: equivalent to the target DFA."
    if witness:
        return False, f"Equivalence check: not equivalent to the target DFA. Counterexample: {witness}"
    return False, "Equivalence check: not equivalent to the target DFA."
