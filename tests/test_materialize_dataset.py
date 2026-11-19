from __future__ import annotations

import pytest

from evidencegraph.final_artifacts import (
    FINAL_COMBINED_VALIDATION,
    FINAL_WIKIDATA_AUDIT,
    FINAL_WIKIDATA_TARGETS,
    FINAL_WIKIDATA_TERMS,
)
from materialize_dataset import parse_args, validated_paraphrase_index


def test_materializer_defaults_to_canonical_final_release_inputs() -> None:
    args = parse_args([])
    assert args.selection.name == "evidencegraph20k_final_selection.jsonl"
    assert args.validation == FINAL_COMBINED_VALIDATION
    assert args.wikidata_targets == FINAL_WIKIDATA_TARGETS
    assert args.wikidata_terms == FINAL_WIKIDATA_TERMS
    assert args.wikidata_audit == FINAL_WIKIDATA_AUDIT


def _selection(corruption: str = "duplication") -> dict:
    return {
        "claim_id": "c1",
        "source_record_id": "source-1",
        "corruption": corruption,
    }


def _validation(paraphrases: list[str], required: int = 1) -> dict:
    return {
        "claim_id": "c1",
        "source_record_id": "source-1",
        "status": "validated",
        "validator_model": "validator",
        "validator_revision": "revision",
        "validator_protocol": "protocol",
        "threshold": 0.8,
        "required_validated": required,
        "paraphrases": paraphrases,
    }


def test_validated_paraphrase_index_binds_identity_and_requirement() -> None:
    index, validator = validated_paraphrase_index(
        [_selection()], [_validation(["Rewrite."])]
    )

    assert index == {"c1": ["Rewrite."]}
    assert validator == ("validator", "revision", "protocol", 0.8)


@pytest.mark.parametrize("field", ["validator_revision", "validator_protocol"])
def test_validated_paraphrase_index_requires_complete_validator_identity(
    field: str,
) -> None:
    record = _validation(["Rewrite."])
    record.pop(field)

    with pytest.raises(ValueError, match="invalid validator provenance"):
        validated_paraphrase_index([_selection()], [record])


def test_validated_paraphrase_index_requires_four_flood_paraphrases() -> None:
    with pytest.raises(ValueError, match="requirement mismatch"):
        validated_paraphrase_index(
            [_selection("paraphrase_flood")], [_validation(["Rewrite."], required=1)]
        )


def test_validated_paraphrase_index_rejects_duplicate_validation_ids() -> None:
    record = _validation(["Rewrite."])

    with pytest.raises(ValueError, match="duplicate validation claim ID"):
        validated_paraphrase_index([_selection()], [record, record])


def test_validated_paraphrase_index_rejects_source_mismatch() -> None:
    record = _validation(["Rewrite."])
    record["source_record_id"] = "other"

    with pytest.raises(ValueError, match="source record mismatch"):
        validated_paraphrase_index([_selection()], [record])
