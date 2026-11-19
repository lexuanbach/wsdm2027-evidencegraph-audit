"""Conservative structural checks for generated evidence paraphrases.

The gate deliberately favors precision over yield.  Sentence-level NLI can
hide a missing predicate or a moved modifier inside an otherwise entailed
sentence, so this module requires the original content lemmas and verbal uses
to survive and separately checks high-risk temporal attachments.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from importlib import metadata
import re
from typing import Any, Iterable


_CONTENT_POS = frozenset({"NOUN", "PROPN", "VERB", "ADJ", "ADV"})
_LIGHT_VERBS = frozenset({"be", "do", "have"})
_HIGH_RISK_MARKERS = frozenset(
    {
        "after",
        "before",
        "during",
        "except",
        "least",
        "less",
        "more",
        "most",
        "only",
        "since",
        "than",
        "throughout",
        "until",
        "without",
        "within",
    }
)
_TEMPORAL_PREPOSITIONS = frozenset(
    {"after", "before", "during", "for", "since", "throughout", "until", "within"}
)
_ASPECT_MODAL_MARKERS = frozenset(
    {"being", "been", "can", "could", "may", "might", "must", "shall", "should", "will", "would"}
)
_BRACKETED_PHONETIC = re.compile(r"\[[^\[\]\n]+\]")
_WHOLE_BRACKET_WRAPPER = re.compile(r"^\[.+\][.!?]?$", re.DOTALL)
_PROMPT_ECHO_MARKERS = (
    '"id":',
    '"items":',
    '"original":',
    '"paraphrase":',
    "content_lemmas",
    "phonetic_spans_copy_exactly",
    "protected_spans_copy_exactly",
    "relation_markers",
    "required_structure_checklist",
    "semantic_roles",
    "temporal_attachments",
)


@dataclass(frozen=True)
class ParsedToken:
    """Parser-neutral token fields needed by the structural gate."""

    index: int
    text: str
    lemma: str
    pos: str
    dependency: str
    head_index: int


@dataclass(frozen=True)
class StructuralGateResult:
    accepted: bool
    reasons: tuple[str, ...]


def _normal_lemma(token: ParsedToken) -> str:
    return token.lemma.strip().casefold()


def _content_signature(tokens: Iterable[ParsedToken]) -> Counter[tuple[str, str]]:
    signature: Counter[tuple[str, str]] = Counter()
    for token in tokens:
        lemma = _normal_lemma(token)
        if token.pos not in _CONTENT_POS or not lemma:
            continue
        if token.pos == "VERB" and lemma in _LIGHT_VERBS:
            continue
        signature[(token.pos, lemma)] += 1
    return signature


def _marker_signature(tokens: Iterable[ParsedToken]) -> Counter[str]:
    return Counter(
        lemma
        for token in tokens
        if (lemma := _normal_lemma(token)) in _HIGH_RISK_MARKERS
    )


def _aspect_modal_signature(tokens: Iterable[ParsedToken]) -> Counter[str]:
    return Counter(
        token.text.strip().casefold()
        for token in tokens
        if token.text.strip().casefold() in _ASPECT_MODAL_MARKERS
    )


def phonetic_span_faithfulness(original: str, candidate: str) -> tuple[bool, list[str]]:
    """Require every explicit bracketed pronunciation to survive verbatim."""

    candidate_spans = Counter(
        " ".join(span.split()).casefold() for span in _BRACKETED_PHONETIC.findall(candidate)
    )
    original_spans = Counter(
        " ".join(span.split()).casefold() for span in _BRACKETED_PHONETIC.findall(original)
    )
    reasons = _missing_counts(original_spans, candidate_spans, "missing_phonetic_span")
    return not reasons, reasons


def surface_form_faithfulness(original: str, candidate: str) -> tuple[bool, list[str]]:
    """Reject schema echoes and annotation wrappers that are not paraphrases."""

    stripped = candidate.strip()
    lowered = stripped.casefold()
    reasons = [
        f"prompt_echo_marker:{marker}"
        for marker in _PROMPT_ECHO_MARKERS
        if marker in lowered
    ]
    if "```" in stripped:
        reasons.append("prompt_echo_code_fence")
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        reasons.append("prompt_echo_serialized_wrapper")
    if _WHOLE_BRACKET_WRAPPER.fullmatch(stripped):
        reasons.append("added_whole_segment_bracket_wrapper")

    original_spans = Counter(
        " ".join(span.split()).casefold()
        for span in _BRACKETED_PHONETIC.findall(original)
    )
    candidate_spans = Counter(
        " ".join(span.split()).casefold()
        for span in _BRACKETED_PHONETIC.findall(candidate)
    )
    reasons.extend(
        _missing_counts(candidate_spans, original_spans, "added_bracketed_span")
    )
    return not reasons, list(dict.fromkeys(reasons))


def _nearest_content_governor(
    tokens: list[ParsedToken], start_index: int
) -> tuple[str, str]:
    by_index = {token.index: token for token in tokens}
    current = by_index[start_index]
    visited: set[int] = set()
    while current.index not in visited:
        visited.add(current.index)
        if current.pos in _CONTENT_POS and _normal_lemma(current):
            return current.pos, _normal_lemma(current)
        if current.head_index == current.index or current.head_index not in by_index:
            break
        current = by_index[current.head_index]
    return "ROOT", "root"


def _temporal_attachments(
    tokens: list[ParsedToken],
) -> Counter[tuple[str, str, str, str]]:
    by_index = {token.index: token for token in tokens}
    attachments: Counter[tuple[str, str, str, str]] = Counter()
    for token in tokens:
        if token.dependency != "pobj" or token.head_index not in by_index:
            continue
        preposition = by_index[token.head_index]
        prep_lemma = _normal_lemma(preposition)
        if prep_lemma not in _TEMPORAL_PREPOSITIONS:
            continue
        governor_pos, governor_lemma = _nearest_content_governor(
            tokens, preposition.head_index
        )
        attachments[
            (_normal_lemma(token), prep_lemma, governor_pos, governor_lemma)
        ] += 1
    return attachments


def _semantic_roles(
    tokens: list[ParsedToken],
) -> Counter[tuple[str, str, str]]:
    """Extract conservative noun/proper-noun roles for explicit predicates."""

    by_index = {token.index: token for token in tokens}

    def role_for(token: ParsedToken) -> tuple[str, str] | None:
        current = token
        visited: set[int] = set()
        while current.index not in visited:
            visited.add(current.index)
            if current.dependency in {"nsubj", "csubj"}:
                head = by_index.get(current.head_index)
                if head is not None and head.pos == "VERB" and _normal_lemma(head) not in _LIGHT_VERBS:
                    return _normal_lemma(head), "agent"
                return None
            if current.dependency in {"nsubjpass", "csubjpass", "dobj", "obj"}:
                head = by_index.get(current.head_index)
                if head is not None and head.pos == "VERB" and _normal_lemma(head) not in _LIGHT_VERBS:
                    return _normal_lemma(head), "patient"
                return None
            if current.dependency == "pobj":
                preposition = by_index.get(current.head_index)
                if preposition is None:
                    return None
                predicate = by_index.get(preposition.head_index)
                if (
                    preposition.dependency == "agent"
                    and predicate is not None
                    and predicate.pos == "VERB"
                    and _normal_lemma(predicate) not in _LIGHT_VERBS
                ):
                    return _normal_lemma(predicate), "agent"
                return None
            if current.dependency not in {"appos", "conj"}:
                return None
            if current.head_index == current.index or current.head_index not in by_index:
                return None
            current = by_index[current.head_index]
        return None

    roles: Counter[tuple[str, str, str]] = Counter()
    for token in tokens:
        if token.pos not in {"NOUN", "PROPN"}:
            continue
        assignment = role_for(token)
        if assignment is None:
            continue
        predicate, role = assignment
        roles[(_normal_lemma(token), predicate, role)] += 1
    return roles


def _missing_counts(
    required: Counter[Any], observed: Counter[Any], prefix: str
) -> list[str]:
    reasons = []
    for key, count in sorted(required.items(), key=lambda item: str(item[0])):
        missing = count - observed[key]
        if missing > 0:
            if isinstance(key, tuple):
                rendered = ":".join(str(value) for value in key)
            else:
                rendered = str(key)
            reasons.append(f"{prefix}:{rendered}:{missing}")
    return reasons


def structural_prompt_constraints(
    original: str, tokens: list[ParsedToken]
) -> dict[str, list[str]]:
    """Render the frozen gate's requirements as a JSON-friendly checklist."""

    def render(counter: Counter[Any]) -> list[str]:
        output = []
        for key, count in sorted(counter.items(), key=lambda item: str(item[0])):
            value = ":".join(str(part) for part in key) if isinstance(key, tuple) else str(key)
            output.append(f"{value}:count={count}")
        return output

    phonetic = Counter(
        " ".join(span.split()) for span in _BRACKETED_PHONETIC.findall(original)
    )
    return {
        "content_lemmas": render(_content_signature(tokens)),
        "relation_markers": render(_marker_signature(tokens)),
        "aspect_modal_markers": render(_aspect_modal_signature(tokens)),
        "semantic_roles": render(_semantic_roles(tokens)),
        "temporal_attachments": render(_temporal_attachments(tokens)),
        "phonetic_spans_copy_exactly": render(phonetic),
    }


def compare_parsed_segments(
    original: list[ParsedToken], candidate: list[ParsedToken]
) -> StructuralGateResult:
    """Check one aligned segment using already parsed tokens."""

    original_content = _content_signature(original)
    candidate_content = _content_signature(candidate)
    reasons = _missing_counts(
        original_content,
        candidate_content,
        "missing_content_lemma",
    )
    reasons.extend(
        _missing_counts(
            candidate_content,
            original_content,
            "added_content_lemma",
        )
    )
    reasons.extend(
        _missing_counts(
            _marker_signature(original),
            _marker_signature(candidate),
            "missing_relation_marker",
        )
    )
    original_aspect = _aspect_modal_signature(original)
    candidate_aspect = _aspect_modal_signature(candidate)
    for marker in sorted(set(original_aspect) | set(candidate_aspect)):
        if original_aspect[marker] != candidate_aspect[marker]:
            reasons.append(
                "changed_aspect_modal_marker:"
                f"{marker}:{original_aspect[marker]}:{candidate_aspect[marker]}"
            )
    reasons.extend(
        _missing_counts(
            _temporal_attachments(original),
            _temporal_attachments(candidate),
            "changed_temporal_attachment",
        )
    )
    reasons.extend(
        _missing_counts(
            _semantic_roles(original),
            _semantic_roles(candidate),
            "changed_semantic_role",
        )
    )
    return StructuralGateResult(accepted=not reasons, reasons=tuple(reasons))


class SpacyLexicallyAnchoredGate:
    """spaCy-backed adapter around the parser-neutral conservative gate."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        import spacy

        self.model_name = model_name
        self.spacy_version = metadata.version("spacy")
        try:
            self.model_version = metadata.version(model_name)
        except metadata.PackageNotFoundError:
            self.model_version = "unknown"
        self.nlp = spacy.load(model_name, disable=["ner"])

    @staticmethod
    def _freeze(document: Any) -> list[ParsedToken]:
        return [
            ParsedToken(
                index=int(token.i),
                text=str(token.text),
                lemma=str(token.lemma_),
                pos=str(token.pos_),
                dependency=str(token.dep_),
                head_index=int(token.head.i),
            )
            for token in document
            if not token.is_space
        ]

    def validate(self, original: str, candidate: str) -> StructuralGateResult:
        original_segments = [line.strip() for line in original.splitlines() if line.strip()]
        candidate_segments = [line.strip() for line in candidate.splitlines() if line.strip()]
        if len(original_segments) != len(candidate_segments):
            return StructuralGateResult(False, ("segment_count_mismatch",))
        reasons: list[str] = []
        for index, (original_segment, candidate_segment) in enumerate(
            zip(original_segments, candidate_segments)
        ):
            _surface_ok, surface_reasons = surface_form_faithfulness(
                original_segment, candidate_segment
            )
            reasons.extend(f"segment_{index}:{reason}" for reason in surface_reasons)
            _phonetic_ok, phonetic_reasons = phonetic_span_faithfulness(
                original_segment, candidate_segment
            )
            reasons.extend(
                f"segment_{index}:{reason}" for reason in phonetic_reasons
            )
            result = compare_parsed_segments(
                self._freeze(self.nlp(original_segment)),
                self._freeze(self.nlp(candidate_segment)),
            )
            reasons.extend(f"segment_{index}:{reason}" for reason in result.reasons)
        return StructuralGateResult(accepted=not reasons, reasons=tuple(reasons))

    def prompt_constraints(self, text: str) -> dict[str, list[dict[str, list[str]]]]:
        """Expose per-segment gate requirements for generator-side self-checking."""
        segments = [line.strip() for line in text.splitlines() if line.strip()]
        return {
            "segments": [
                structural_prompt_constraints(
                    segment, self._freeze(self.nlp(segment))
                )
                for segment in segments
            ]
        }
