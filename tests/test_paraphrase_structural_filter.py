from __future__ import annotations

from evidencegraph.paraphrase_structural_filter import filter_validation_record
from evidencegraph.structural_validation import StructuralGateResult


class FakeGate:
    def validate(self, original: str, candidate: str) -> StructuralGateResult:
        if candidate == "Faithful rewrite.":
            return StructuralGateResult(True, ())
        return StructuralGateResult(False, ("missing_content_lemma:VERB:develop:1",))


def base_record() -> dict:
    return {
        "claim_id": "c1",
        "source_record_id": "s1",
        "generator_model": "generator",
        "prompt_version": "prompt",
        "validator_model": "nli",
        "validator_revision": "revision",
        "validator_protocol": "protocol",
        "threshold": 0.8,
        "required_validated": 1,
        "original": "Original evidence.",
        "paraphrases": ["Altered rewrite."],
        "n_candidates": 2,
        "n_accepted": 2,
        "candidate_results": [
            {
                "candidate": "Altered rewrite.",
                "accepted": True,
                "forward_entailment": 0.9,
                "reverse_entailment": 0.9,
                "reasons": [],
            },
            {
                "candidate": "Faithful rewrite.",
                "accepted": True,
                "forward_entailment": 0.9,
                "reverse_entailment": 0.9,
                "reasons": [],
            },
        ],
        "status": "validated",
    }


def test_structural_filter_reselects_first_candidate_that_passes_both_gates() -> None:
    output = filter_validation_record(
        base_record(),
        FakeGate(),
        composite_validator_id="composite",
        parser_model="parser",
        parser_model_version="1",
        spacy_version="2",
        source_validation_sha256="a" * 64,
    )
    assert output["status"] == "validated"
    assert output["paraphrases"] == ["Faithful rewrite."]
    assert output["n_accepted"] == 1
    assert not output["candidate_results"][0]["accepted"]
    assert output["candidate_results"][1]["accepted"]
    assert output["nli_validator_model"] == "nli"
    assert output["nli_validator_revision"] == "revision"
    assert output["nli_validator_protocol"] == "protocol"


def test_structural_filter_preserves_base_nli_rejection_without_running_gate() -> None:
    record = base_record()
    record["candidate_results"][1]["accepted"] = False
    output = filter_validation_record(
        record,
        FakeGate(),
        composite_validator_id="composite",
        parser_model="parser",
        parser_model_version="1",
        spacy_version="2",
        source_validation_sha256="a" * 64,
    )
    second = output["candidate_results"][1]
    assert second["structural_accepted"] is None
    assert second["structural_reasons"] == ["not_evaluated_base_nli_rejection"]
