from __future__ import annotations

import json

import pytest

from evidencegraph.human_audit_package import (
    build_audit_package,
    validate_audit_package,
)


def _episode() -> dict:
    return {
        "claim_id": "claim-1",
        "claim": "A test claim.",
        "split": "test",
        "label": True,
        "corruption": "paraphrase_flood",
        "sources": [
            {
                "source_id": "claim-1-mirror-0",
                "text": "First visible source.",
                "family": "mirror",
                "stance": 1.0,
                "stance_confidence": 0.95,
                "domain": "mirror-0.synthetic.invalid",
                "origin_id": "origin-1",
                "cited_source_ids": [],
                "synthetic": True,
                "metadata": {},
            },
            {
                "source_id": "claim-1-citation-0",
                "text": "Second visible source cites the first.",
                "family": "citation",
                "stance": 1.0,
                "stance_confidence": 0.95,
                "domain": "citation.synthetic.invalid",
                "origin_id": "origin-1",
                "cited_source_ids": ["claim-1-mirror-0"],
                "synthetic": True,
                "metadata": {},
            },
        ],
    }


def test_audit_packet_hides_construction_fields_and_source_ids() -> None:
    tasks, source_key = build_audit_package([_episode()])
    serialized = json.dumps(tasks)

    assert "claim-1-mirror-0" not in serialized
    assert "claim-1-citation-0" not in serialized
    for forbidden in ("family", "stance", "domain", "origin_id", "synthetic"):
        assert f'"{forbidden}"' not in serialized
    assert tasks[0]["sources"][1]["cited_source_ids"] or tasks[0]["sources"][0][
        "cited_source_ids"
    ]
    claim_ids, source_ids = validate_audit_package(
        [_episode()], tasks, source_key
    )
    assert claim_ids == {"claim-1"}
    assert len(source_ids) == 2


def test_audit_packet_verification_rejects_visible_text_change() -> None:
    tasks, source_key = build_audit_package([_episode()])
    tasks[0]["sources"][0]["text"] = "Changed after packet preparation."

    with pytest.raises(ValueError, match="do not match"):
        validate_audit_package([_episode()], tasks, source_key)
