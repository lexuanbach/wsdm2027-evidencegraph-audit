from pathlib import Path

from evidencegraph.dataset_builder import write_jsonl
from make_paraphrase_retry import excluded_claim_ids


def test_excluded_claim_ids_unions_multiple_attempted_tiers(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_jsonl([{"claim_id": "a"}, {"claim_id": "b"}], first)
    write_jsonl([{"claim_id": "b"}, {"claim_id": "c"}], second)

    assert excluded_claim_ids([first, second]) == {"a", "b", "c"}
