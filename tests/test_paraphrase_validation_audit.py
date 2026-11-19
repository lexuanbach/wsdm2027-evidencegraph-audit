from __future__ import annotations

import json

from evidencegraph.paraphrase_validation_audit import audit_validation_tranche


def test_audit_counts_structural_rejection_reasons(tmp_path) -> None:
    selection = tmp_path / "selection.jsonl"
    validation = tmp_path / "validation.jsonl"
    selection.write_text(
        json.dumps(
            {
                "claim_id": "c1",
                "corruption": "entity_swap",
                "split": "dev",
                "label": "SUPPORTED",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    validation.write_text(
        json.dumps(
            {
                "claim_id": "c1",
                "validator_model": "nli+structure-v3",
                "validator_revision": "revision",
                "validator_protocol": "protocol",
                "threshold": 0.8,
                "status": "semantic_failure",
                "required_validated": 1,
                "original": "The group developed the album.",
                "candidate_results": [
                    {
                        "candidate": "The group released the album.",
                        "accepted": False,
                        "forward_entailment": 0.91,
                        "reverse_entailment": 0.92,
                        "reasons": [],
                        "structural_reasons": [
                            "missing_content_lemma:VERB:develop:1"
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audit = audit_validation_tranche(selection, validation, project_root=tmp_path)

    assert audit["rejection_category_candidate_counts"] == {
        "structural_content_lemma_missing": 1
    }
    assert audit["top_exact_reasons"] == {
        "missing_content_lemma:VERB:develop:1": 1
    }
    assert audit["validator_revision"] == "revision"
    assert audit["validator_protocol"] == "protocol"
