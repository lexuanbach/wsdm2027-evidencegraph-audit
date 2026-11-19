from __future__ import annotations

import numpy as np

from evidencegraph.semantic_validation import (
    MutualEntailmentValidator,
    lexical_faithfulness,
    protected_spans,
)


def test_protected_spans_include_entities_and_numbers() -> None:
    spans = protected_spans("The Apollo 11 mission reached the Moon in 1969.")
    assert {"Apollo", "Moon", "11", "1969"}.issubset(spans)


def test_number_guard_does_not_capture_trailing_punctuation() -> None:
    assert "1988" in protected_spans("In 1988, an event occurred.")
    assert "1988," not in protected_spans("In 1988, an event occurred.")


def test_lexical_gate_rejects_changed_entity() -> None:
    accepted, reasons = lexical_faithfulness(
        "A quokka is native to Australia.", "A koala is native to New Zealand."
    )
    assert not accepted
    assert "missing:Australia" in reasons


def test_lexical_gate_accepts_conservative_rewording() -> None:
    accepted, reasons = lexical_faithfulness(
        "Apollo 11 reached the Moon in 1969.", "In 1969, Apollo 11 arrived at the Moon."
    )
    assert accepted
    assert reasons == []


def test_batched_record_validation_preserves_record_boundaries() -> None:
    validator = MutualEntailmentValidator.__new__(MutualEntailmentValidator)
    validator.threshold = 0.8
    validator._entailment = lambda premises, hypotheses, batch_size: np.full(  # type: ignore[method-assign]
        len(premises), 0.9
    )

    results = validator.validate_records(
        [
            ("Apollo reached the Moon.", ["Apollo arrived at the Moon."]),
            (
                "Sansa Stark is played by Sophie Turner.",
                [
                    "Sophie Turner plays Sansa Stark.",
                    "Sansa Stark is portrayed by Sophie Turner.",
                ],
            ),
        ],
        batch_size=2,
    )

    assert [len(group) for group in results] == [1, 2]
    assert all(result.accepted for group in results for result in group)


def test_segmented_validation_requires_every_aligned_segment() -> None:
    validator = MutualEntailmentValidator.__new__(MutualEntailmentValidator)
    validator.threshold = 0.8
    validator._entailment = lambda premises, hypotheses, batch_size: np.full(  # type: ignore[method-assign]
        len(premises), 0.9
    )

    results = validator.validate_records(
        [
            (
                "Apollo reached the Moon.\nSansa Stark met Sophie Turner.",
                [
                    "Apollo arrived at the Moon.\nSophie Turner met Sansa Stark.",
                    "Apollo arrived at the Moon.",
                ],
            )
        ]
    )[0]

    assert results[0].accepted
    assert not results[1].accepted
    assert results[1].reasons == ("segment_count_mismatch",)
