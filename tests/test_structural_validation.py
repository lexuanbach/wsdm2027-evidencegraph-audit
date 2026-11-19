from __future__ import annotations

from evidencegraph.structural_validation import (
    ParsedToken,
    compare_parsed_segments,
    phonetic_span_faithfulness,
    surface_form_faithfulness,
    structural_prompt_constraints,
)


def tok(
    index: int,
    text: str,
    lemma: str,
    pos: str,
    dependency: str,
    head_index: int,
) -> ParsedToken:
    return ParsedToken(index, text, lemma, pos, dependency, head_index)


def test_gate_rejects_omitted_content_predicate() -> None:
    original = [
        tok(0, "divisions", "division", "NOUN", "nsubj", 1),
        tok(1, "develop", "develop", "VERB", "ROOT", 1),
        tok(2, "produce", "produce", "VERB", "conj", 1),
        tok(3, "projects", "project", "NOUN", "dobj", 2),
    ]
    candidate = [
        tok(0, "divisions", "division", "NOUN", "nsubj", 1),
        tok(1, "produce", "produce", "VERB", "ROOT", 1),
        tok(2, "projects", "project", "NOUN", "dobj", 1),
    ]
    result = compare_parsed_segments(original, candidate)
    assert not result.accepted
    assert "missing_content_lemma:VERB:develop:1" in result.reasons


def test_gate_rejects_added_or_duplicated_content() -> None:
    original = [
        tok(0, "film", "film", "NOUN", "nsubj", 1),
        tok(1, "stars", "star", "VERB", "ROOT", 1),
    ]
    candidate = [
        tok(0, "film", "film", "NOUN", "nsubj", 2),
        tok(1, "film", "film", "NOUN", "compound", 0),
        tok(2, "stars", "star", "VERB", "ROOT", 2),
    ]
    result = compare_parsed_segments(original, candidate)
    assert not result.accepted
    assert "added_content_lemma:NOUN:film:1" in result.reasons


def test_gate_rejects_changed_temporal_attachment() -> None:
    original = [
        tok(0, "active", "active", "ADJ", "advcl", 4),
        tok(1, "for", "for", "ADP", "prep", 0),
        tok(2, "decade", "decade", "NOUN", "pobj", 1),
        tok(3, "he", "he", "PRON", "nsubj", 4),
        tok(4, "played", "play", "VERB", "ROOT", 4),
    ]
    candidate = [
        tok(0, "he", "he", "PRON", "nsubj", 1),
        tok(1, "played", "play", "VERB", "ROOT", 1),
        tok(2, "active", "active", "ADJ", "acomp", 1),
        tok(3, "for", "for", "ADP", "prep", 1),
        tok(4, "decade", "decade", "NOUN", "pobj", 3),
    ]
    result = compare_parsed_segments(original, candidate)
    assert not result.accepted
    assert any(reason.startswith("changed_temporal_attachment:decade:for") for reason in result.reasons)


def test_gate_accepts_active_passive_reordering_with_same_content() -> None:
    original = [
        tok(0, "Wikipedia", "Wikipedia", "PROPN", "nsubjpass", 2),
        tok(1, "was", "be", "AUX", "auxpass", 2),
        tok(2, "launched", "launch", "VERB", "ROOT", 2),
        tok(3, "Jimmy", "Jimmy", "PROPN", "pobj", 2),
    ]
    candidate = [
        tok(0, "Jimmy", "Jimmy", "PROPN", "nsubj", 1),
        tok(1, "launched", "launch", "VERB", "ROOT", 1),
        tok(2, "Wikipedia", "Wikipedia", "PROPN", "dobj", 1),
    ]
    result = compare_parsed_segments(original, candidate)
    assert result.accepted
    assert result.reasons == ()


def test_gate_rejects_album_title_relation_reversal() -> None:
    original = [
        tok(0, "album", "album", "NOUN", "nsubjpass", 3),
        tok(1, "Suburbs", "Suburbs", "PROPN", "appos", 0),
        tok(2, "was", "be", "AUX", "auxpass", 3),
        tok(3, "released", "release", "VERB", "ROOT", 3),
    ]
    candidate = [
        tok(0, "Suburbs", "Suburbs", "PROPN", "nsubjpass", 2),
        tok(1, "album", "album", "NOUN", "pobj", 3),
        tok(2, "released", "release", "VERB", "ROOT", 2),
        tok(3, "by", "by", "ADP", "agent", 2),
    ]
    result = compare_parsed_segments(original, candidate)
    assert not result.accepted
    assert "changed_semantic_role:album:release:patient:1" in result.reasons


def test_gate_rejects_omitted_pronunciation() -> None:
    accepted, reasons = phonetic_span_faithfulness(
        "David ([dəˈviːd]) is an actor.", "David is an actor."
    )
    assert not accepted
    assert reasons == ["missing_phonetic_span:[dəˈviːd]:1"]


def test_surface_gate_rejects_prompt_echoes_and_added_brackets() -> None:
    original = "The film stars Smith."
    for candidate in (
        '[The film stars Smith.].',
        '{"id": "c::segment-0", "original": "The film stars Smith."}',
        "[The] film stars Smith.",
    ):
        accepted, reasons = surface_form_faithfulness(original, candidate)
        assert not accepted
        assert reasons


def test_surface_gate_allows_preserved_source_phonetic_brackets() -> None:
    accepted, reasons = surface_form_faithfulness(
        "David ([dəˈviːd]) is an actor.",
        "An actor is what David ([dəˈviːd]) is.",
    )
    assert accepted
    assert reasons == []


def test_gate_rejects_added_progressive_aspect() -> None:
    original = [
        tok(0, "film", "film", "NOUN", "nsubjpass", 1),
        tok(1, "directed", "direct", "VERB", "ROOT", 1),
    ]
    candidate = [
        tok(0, "film", "film", "NOUN", "nsubjpass", 2),
        tok(1, "being", "be", "AUX", "auxpass", 2),
        tok(2, "directed", "direct", "VERB", "ROOT", 2),
    ]
    result = compare_parsed_segments(original, candidate)
    assert not result.accepted
    assert "changed_aspect_modal_marker:being:0:1" in result.reasons


def test_prompt_constraints_render_content_and_roles_deterministically() -> None:
    tokens = [
        tok(0, "Uruguay", "Uruguay", "PROPN", "nsubj", 2),
        tok(1, "also", "also", "ADV", "advmod", 2),
        tok(2, "won", "win", "VERB", "ROOT", 2),
    ]

    constraints = structural_prompt_constraints("Uruguay also won.", tokens)

    assert constraints["content_lemmas"] == [
        "ADV:also:count=1",
        "PROPN:uruguay:count=1",
        "VERB:win:count=1",
    ]
    assert constraints["semantic_roles"] == ["uruguay:win:agent:count=1"]
