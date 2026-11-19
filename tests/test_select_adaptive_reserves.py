from __future__ import annotations

import pytest

from select_adaptive_reserves import adaptive_targets


def _primary(claim_id: str, corruption: str) -> dict:
    return {
        "claim_id": claim_id,
        "split": "train",
        "fever_label": "SUPPORTS",
        "corruption": corruption,
    }


def test_adaptive_targets_scale_each_observed_failure() -> None:
    primary = [_primary("a", "duplication"), _primary("b", "paraphrase_flood")]
    validation = [
        {"claim_id": "a", "status": "semantic_failure"},
        {"claim_id": "b", "status": "validated"},
    ]

    failures, targets = adaptive_targets(primary, validation, per_failure=2)

    key = ("train", "SUPPORTS", "duplication")
    assert failures == {key: 1}
    assert targets == {key: 2}


def test_adaptive_targets_require_exact_primary_coverage() -> None:
    with pytest.raises(ValueError, match="exactly the complete primary validation"):
        adaptive_targets(
            [_primary("a", "duplication")],
            [{"claim_id": "other", "status": "validated"}],
            per_failure=2,
        )
