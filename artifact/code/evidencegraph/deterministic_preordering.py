"""Deterministic, lexically lossless constituent preordering for hard cases."""

from __future__ import annotations

import re
from typing import Any, Protocol

from .paraphrase_job import (
    DETERMINISTIC_COORDINATION_PROMPT_VERSION,
    DETERMINISTIC_EXPANDED_SYNTAX_PROMPT_VERSION,
    DETERMINISTIC_PREORDER_PROMPT_VERSION,
    DETERMINISTIC_PSEUDOCLEFT_PROMPT_VERSION,
)
from .semantic_validation import lexical_faithfulness


MODEL_ID = "deterministic-rules:syntactic-preordering-v17"
PROMPT_VERSION = DETERMINISTIC_PREORDER_PROMPT_VERSION
COORDINATION_MODEL_ID = "deterministic-rules:syntactic-coordination-v18"
COORDINATION_PROMPT_VERSION = DETERMINISTIC_COORDINATION_PROMPT_VERSION
PSEUDOCLEFT_MODEL_ID = "deterministic-rules:copular-pseudocleft-v19"
PSEUDOCLEFT_PROMPT_VERSION = DETERMINISTIC_PSEUDOCLEFT_PROMPT_VERSION
EXPANDED_MODEL_ID = "deterministic-rules:dependency-genitive-reordering-v20"
EXPANDED_PROMPT_VERSION = DETERMINISTIC_EXPANDED_SYNTAX_PROMPT_VERSION

_LEADING_ADJUNCT = re.compile(
    r"^(In|On|At|By|From|During|After|Before|Following|Since|Between|Within|"
    r"Throughout|As|While|When|Although|Though|Because|Despite|According to|"
    r"Originally|Initially|Subsequently|Later|Previously|Currently|Ultimately|"
    r"Eventually|Together with)\b([^,]{0,100}),\s+(.+?)([.!?])$",
    re.IGNORECASE,
)
_MONTH_DATE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(\d{1,2}),\s+(\d{4})\b"
)
_PARENTHETICAL = re.compile(r"\s*(\([^()]{2,100}\))")
_TERMINAL_ADJUNCT = re.compile(
    r"^(.+?)(?:,)?\s+(in|on|at|during|after|before|since|within|throughout)\s+"
    r"([^,.!?]{2,80})([.!?])$",
    re.IGNORECASE,
)
_MOVABLE_DEPENDENCIES = frozenset({"prep"})
_TEMPORAL_PREPOSITIONS = frozenset(
    {"after", "before", "during", "in", "on", "since", "throughout", "until", "within"}
)
_MONTHS = frozenset(
    {
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
    }
)


class GateResult(Protocol):
    accepted: bool


class StructuralGate(Protocol):
    def validate(self, original: str, candidate: str) -> GateResult: ...


def candidate_transformations(sentence: str) -> list[str]:
    """Return deterministic surface reorderings without lexical substitution."""

    outputs: list[str] = []
    leading = _LEADING_ADJUNCT.match(sentence)
    if leading:
        adjunct = (leading.group(1) + leading.group(2)).strip()
        rest = leading.group(3).strip()
        adjunct = adjunct[0].lower() + adjunct[1:]
        outputs.append(f"{rest}, {adjunct}{leading.group(4)}")

    date = _MONTH_DATE.search(sentence)
    if date:
        replacement = f"{date.group(2)} {date.group(1)} {date.group(3)}"
        outputs.append(sentence[: date.start()] + replacement + sentence[date.end() :])

    parenthetical = _PARENTHETICAL.search(sentence)
    if parenthetical:
        core = (sentence[: parenthetical.start()] + sentence[parenthetical.end() :]).strip()
        core = re.sub(r"\s+([,.!?])", r"\1", core)
        core = re.sub(r"\s{2,}", " ", core)
        phrase = parenthetical.group(1)
        if core and core[-1] in ".!?":
            outputs.append(f"{core[:-1].rstrip()} {phrase}{core[-1]}")
        else:
            outputs.append(f"{core} {phrase}")

    trailing = _TERMINAL_ADJUNCT.match(sentence)
    if trailing:
        base = trailing.group(1).rstrip(" ,")
        adjunct = f"{trailing.group(2)} {trailing.group(3)}"
        adjunct = adjunct[0].upper() + adjunct[1:]
        outputs.append(f"{adjunct}, {base}{trailing.group(4)}")

    unique: list[str] = []
    seen = {sentence.casefold().strip()}
    for output in outputs:
        normalized = output.casefold().strip()
        if normalized and normalized not in seen:
            unique.append(output)
            seen.add(normalized)
    return unique


def coordination_transformations(sentence: str, nlp: Any) -> list[str]:
    """Rotate intact nominal conjunct spans while retaining their conjunction."""

    document = nlp(sentence)
    outputs: list[str] = []
    for token in document:
        if token.dep_ == "conj" or not any(
            child.dep_ == "conj" for child in token.children
        ):
            continue
        members: list[Any] = []
        queue = [token]
        seen: set[int] = set()
        while queue:
            member = queue.pop(0)
            if int(member.i) in seen:
                continue
            seen.add(int(member.i))
            members.append(member)
            queue.extend(child for child in member.children if child.dep_ == "conj")
        members.sort(key=lambda item: int(item.i))
        if len(members) < 2 or any(
            member.pos_ not in {"NOUN", "PROPN", "ADJ"} for member in members
        ):
            continue

        spans: list[tuple[int, int, str]] = []
        for member in members:
            indices: list[int] = []
            stack = [member]
            while stack:
                current = stack.pop()
                if int(current.i) in indices:
                    continue
                indices.append(int(current.i))
                stack.extend(
                    child
                    for child in current.children
                    if child.dep_ not in {"conj", "cc", "punct"}
                )
            low, high = min(indices), max(indices)
            if set(indices) != set(range(low, high + 1)):
                spans = []
                break
            spans.append((low, high, document[low : high + 1].text))
        if len(spans) != len(members):
            continue
        between = [
            document[index]
            for index in range(spans[0][0], spans[-1][1] + 1)
            if not any(low <= index <= high for low, high, _text in spans)
        ]
        if any(item.dep_ not in {"punct", "cc"} for item in between):
            continue
        conjunctions = [item.text for item in between if item.dep_ == "cc"]
        conjunction = conjunctions[-1] if conjunctions else "and"
        texts = [span[2] for span in spans]
        rotated = texts[1:] + texts[:1]
        replacement = (
            f" {conjunction} ".join(rotated)
            if len(rotated) == 2
            else ", ".join(rotated[:-1]) + f", {conjunction} {rotated[-1]}"
        )
        first_token = document[spans[0][0]]
        last_token = document[spans[-1][1]]
        stop = int(last_token.idx) + len(str(last_token.text))
        outputs.append(sentence[: int(first_token.idx)] + replacement + sentence[stop:])
    return list(dict.fromkeys(output for output in outputs if output != sentence))


def _clean_join(left: str, right: str) -> str:
    value = f"{left.rstrip()} {right.lstrip()}".strip()
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"^[,;:]\s*", "", value)
    value = re.sub(r",\s*([.!?])$", r"\1", value)
    return re.sub(r"\s{2,}", " ", value)


def dependency_adjunct_transformations(sentence: str, nlp: Any) -> list[str]:
    """Move one simple root-level calendar PP while retaining every token.

    The deliberately narrow rule excludes durations, participial clauses, and
    PPs separated from the root by another verb.  Those cases can be parsed as
    root dependents even when they semantically modify a title, complement, or
    list item.
    """

    if ";" in sentence or ":" in sentence:
        return []
    document = nlp(sentence)
    roots = [token for token in document if token.dep_ == "ROOT"]
    if len(roots) != 1:
        return []
    root = roots[0]
    terminal = sentence[-1] if sentence and sentence[-1] in ".!?" else "."
    outputs = []
    for child in root.children:
        if child.dep_ not in _MOVABLE_DEPENDENCIES:
            continue
        subtree = list(child.subtree)
        indices = sorted(int(token.i) for token in subtree if not token.is_punct)
        if not indices or indices != list(range(indices[0], indices[-1] + 1)):
            continue
        has_calendar_value = any(
            re.fullmatch(r"(?:1\d{3}|20\d{2})", token.text) is not None
            or token.text.casefold() in _MONTHS
            for token in subtree
        )
        if (
            child.lemma_.casefold() not in _TEMPORAL_PREPOSITIONS
            or not has_calendar_value
            or root.lemma_.casefold() == "be"
            or len(indices) > 5
            or any(token.dep_ in {"cc", "conj"} for token in subtree)
        ):
            continue
        low_between = min(int(root.i), indices[0]) + 1
        high_between = max(int(root.i), indices[-1])
        if any(
            document[index].pos_ in {"VERB", "AUX"}
            for index in range(low_between, high_between)
        ):
            continue
        start_char = int(document[indices[0]].idx)
        end_token = document[indices[-1]]
        end_char = int(end_token.idx) + len(str(end_token.text))
        if sentence[:start_char].rstrip().endswith(",") and sentence[end_char:].lstrip().startswith(","):
            continue
        start_token, end_token = document[indices[0]], document[indices[-1]]
        start = int(start_token.idx)
        end = int(end_token.idx) + len(str(end_token.text))
        fragment = sentence[start:end].strip(" ,")
        if not fragment or len(fragment.split()) > 18:
            continue
        core = _clean_join(sentence[:start], sentence[end:]).rstrip(" .!?")
        if not core:
            continue
        if int(child.i) < int(root.i):
            core = core[0].upper() + core[1:]
            moved = fragment[0].lower() + fragment[1:]
            candidate = f"{core}, {moved}{terminal}"
        else:
            moved = fragment[0].upper() + fragment[1:]
            first = next((token for token in document if not token.is_punct), None)
            if first is not None and first.pos_ != "PROPN":
                core = core[0].lower() + core[1:]
            candidate = f"{moved}, {core}{terminal}"
        outputs.append(candidate)
    return list(dict.fromkeys(output for output in outputs if output != sentence))


def possessive_transformations(sentence: str, nlp: Any) -> list[str]:
    """Rewrite a contiguous Saxon genitive as an ``of`` construction."""

    document = nlp(sentence)
    outputs = []
    for possessor in document:
        if (
            possessor.dep_ != "poss"
            or possessor.pos_ == "PRON"
            or possessor.head.pos_ not in {"NOUN", "PROPN"}
            or not possessor.head.text[:1].islower()
        ):
            continue
        owner_tokens = sorted(possessor.subtree, key=lambda token: int(token.i))
        owner_indices = [int(token.i) for token in owner_tokens]
        head_index = int(possessor.head.i)
        if not owner_indices or max(owner_indices) >= head_index:
            continue
        span_indices = list(range(min(owner_indices), head_index + 1))
        if any(document[index].is_punct for index in span_indices):
            continue
        between = [document[index] for index in range(max(owner_indices) + 1, head_index + 1)]
        if not between or any(token.pos_ not in {"DET", "ADJ", "NOUN", "PROPN"} for token in between):
            continue
        start_token = document[min(owner_indices)]
        end_token = document[head_index]
        start = int(start_token.idx)
        end = int(end_token.idx) + len(str(end_token.text))
        owner_start = int(start_token.idx)
        owner_end_token = document[max(owner_indices)]
        owner_end = int(owner_end_token.idx) + len(str(owner_end_token.text))
        owner = sentence[owner_start:owner_end].strip()
        if re.search(r"(?:['’]s|['’])$", owner) is None:
            continue
        owner = re.sub(r"(?:['’]s|['’])$", "", owner).strip()
        if owner.casefold().startswith(("the ", "a ", "an ")):
            owner = owner[0].lower() + owner[1:]
        possessed = document[max(owner_indices) + 1 : head_index + 1].text.strip()
        if not owner or not possessed:
            continue
        article = "The" if start == 0 else "the"
        replacement = f"{article} {possessed} of {owner}"
        outputs.append(_clean_join(sentence[:start], replacement + sentence[end:]))
    return list(dict.fromkeys(output for output in outputs if output != sentence))


def expanded_syntax_transformations(sentence: str, nlp: Any) -> list[str]:
    """Enumerate only templates not already covered by frozen v17--v19."""

    historical = {
        value.casefold().strip()
        for value in (
            *candidate_transformations(sentence),
            *coordination_transformations(sentence, nlp),
            *copular_pseudocleft_transformations(sentence, nlp),
        )
    }
    novel = [
        *dependency_adjunct_transformations(sentence, nlp),
        *possessive_transformations(sentence, nlp),
    ]
    return list(
        dict.fromkeys(
            value
            for value in novel
            if value.casefold().strip() not in historical
        )
    )


def copular_pseudocleft_transformations(sentence: str, nlp: Any) -> list[str]:
    """Form a conservative reverse pseudo-cleft from a simple copular clause.

    The rule rejects punctuation-bearing subject or complement spans because
    moving appositives and relative clauses can change their attachment.
    Content words are retained exactly; only the function word ``what`` and a
    repeated surface copula are introduced.
    """

    document = nlp(sentence)
    roots = [token for token in document if token.dep_ == "ROOT"]
    if len(roots) != 1:
        return []
    root = roots[0]
    if root.lemma_ != "be" or root.text.casefold() not in {"is", "are", "was", "were"}:
        return []
    if any(child.dep_ == "neg" for child in root.children):
        return []
    subjects = [child for child in root.children if child.dep_ in {"nsubj", "csubj"}]
    complements = [
        child for child in root.children if child.dep_ in {"attr", "acomp", "oprd"}
    ]
    if len(subjects) != 1 or len(complements) != 1:
        return []

    subject_indices = sorted(int(token.i) for token in subjects[0].subtree)
    complement_indices = sorted(int(token.i) for token in complements[0].subtree)
    if not subject_indices or not complement_indices:
        return []
    subject_low, subject_high = subject_indices[0], subject_indices[-1]
    complement_low, complement_high = complement_indices[0], complement_indices[-1]
    last_content = max(int(token.i) for token in document if not token.is_punct)
    if not (
        subject_low == 0
        and subject_high < int(root.i) < complement_low
        and complement_high == last_content
    ):
        return []
    if subject_indices != list(range(subject_low, subject_high + 1)):
        return []
    if complement_indices != list(range(complement_low, complement_high + 1)):
        return []
    if any(document[index].is_punct for index in subject_indices + complement_indices):
        return []

    subject = document[subject_low : subject_high + 1].text
    complement = document[complement_low : complement_high + 1].text
    if document[subject_low].pos_ in {"DET", "PRON"}:
        subject = subject[0].lower() + subject[1:]
    complement = complement[0].upper() + complement[1:]
    punctuation = sentence[-1] if sentence and sentence[-1] in ".!?" else "."
    return [f"{complement} {root.text} what {subject} {root.text}{punctuation}"]


def structurally_safe_candidates(
    original: str, gate: StructuralGate, *, include_coordination: bool = False
) -> list[str] | None:
    """Build two candidates when every aligned segment has a safe reordering."""

    segments = [segment.strip() for segment in original.splitlines() if segment.strip()]
    nlp = getattr(gate, "nlp", None)
    candidate_lists = []
    for segment in segments:
        candidates = candidate_transformations(segment)
        if include_coordination and nlp is not None:
            candidates.extend(coordination_transformations(segment, nlp))
        candidate_lists.append(list(dict.fromkeys(candidates)))
    safe_by_segment = [
        [candidate for candidate in candidates if gate.validate(segment, candidate).accepted]
        for segment, candidates in zip(segments, candidate_lists)
    ]
    if not segments or any(not candidates for candidates in safe_by_segment):
        return None
    first = "\n".join(candidates[0] for candidates in safe_by_segment)
    second = "\n".join(candidates[-1] for candidates in safe_by_segment)
    if second.casefold().strip() == first.casefold().strip():
        # The ordinary tier requires two generated candidates but accepts one.
        # Retain the exact original as an explicit validator-rejected fallback.
        second = original
    return [first, second]


def structurally_safe_pseudocleft_candidates(
    original: str, gate: StructuralGate
) -> list[str] | None:
    """Build the v19 candidate pair using only the pseudo-cleft rule."""

    nlp = getattr(gate, "nlp", None)
    if nlp is None:
        return None
    segments = [segment.strip() for segment in original.splitlines() if segment.strip()]
    safe_by_segment: list[list[str]] = []
    for segment in segments:
        candidates = copular_pseudocleft_transformations(segment, nlp)
        safe_by_segment.append(
            [
                candidate
                for candidate in candidates
                if gate.validate(segment, candidate).accepted
            ]
        )
    if not segments or any(not candidates for candidates in safe_by_segment):
        return None
    first = "\n".join(candidates[0] for candidates in safe_by_segment)
    return [first, original]


def structurally_safe_expanded_candidates(
    original: str, gate: StructuralGate
) -> list[str] | None:
    """Build a v20 pair only when every segment has a gate-safe template."""

    nlp = getattr(gate, "nlp", None)
    if nlp is None:
        return None
    segments = [segment.strip() for segment in original.splitlines() if segment.strip()]
    safe_by_segment = []
    for segment in segments:
        safe = [
            candidate
            for candidate in expanded_syntax_transformations(segment, nlp)
            if lexical_faithfulness(segment, candidate)[0]
            and gate.validate(segment, candidate).accepted
        ]
        safe_by_segment.append(safe)
    if not segments or any(not candidates for candidates in safe_by_segment):
        return None
    first = "\n".join(candidates[0] for candidates in safe_by_segment)
    second = "\n".join(candidates[-1] for candidates in safe_by_segment)
    if second.casefold().strip() == first.casefold().strip():
        second = original
    return [first, second]
