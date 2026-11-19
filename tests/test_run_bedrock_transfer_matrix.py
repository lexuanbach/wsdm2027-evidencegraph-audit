from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidencegraph.result_registry import source_descriptor
from run_bedrock_transfer_matrix import (
    MODEL,
    transfer_comparison_command,
    transfer_comparison_path,
    transfer_command,
    transfer_paths,
    unresolved_backend_failure,
    verify_bedrock_connectivity,
    verify_dataset_release_gate,
)


class _Catalog:
    def list_foundation_models(self, **_kwargs):
        return {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "modelSummaries": [{"modelLifecycle": {"status": "ACTIVE"}}],
        }

    def list_inference_profiles(self, **_kwargs):
        return {
            "inferenceProfileSummaries": [
                {"inferenceProfileId": MODEL, "status": "ACTIVE"}
            ]
        }


class _Runtime:
    def converse(self, **_kwargs):
        return {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "output": {"message": {"content": [{"text": "OK"}]}},
        }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_transfer_commands_use_exact_profile_and_graph_paths() -> None:
    fixed = transfer_command("hover_dev", "fixed_rag")
    graph = transfer_command("hover_dev", "evidencegraph")
    expert = transfer_command("expertqa_domain_test", "evidencegraph")
    assert MODEL in fixed
    assert "--backend" in fixed and "bedrock" in fixed
    assert any(item.endswith("hover_dev_evidencegraph_selection.jsonl") for item in graph)
    assert any(item.endswith("run_expertqa_evidencegraph.py") for item in expert)
    assert transfer_paths("fever_dev", "fixed_rag")["quality"].name == (
        "fever_dev_fixed_rag_haiku45_quality.json"
    )
    comparison = transfer_comparison_command("fever_dev")
    assert "10000" in comparison
    assert transfer_comparison_path("fever_dev").name == (
        "fever_dev_evidencegraph_vs_fixed_rag_haiku45_paired.json"
    )


def test_transfer_preflight_contains_no_secret_and_requires_exact_ok() -> None:
    result = verify_bedrock_connectivity(_Catalog(), _Runtime())
    assert result["nova_http_status"] == 200
    assert result["nova_exact_ok"]
    assert result["haiku_profile"] == MODEL
    assert "token" not in json.dumps(result).casefold()


def test_unresolved_backend_failure_is_recoverable_after_completion(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    base = {"dataset": "HoVer", "case_id": "1"}
    _write_jsonl(output, [{**base, "status": "backend_failure"}])
    assert unresolved_backend_failure(output)
    _write_jsonl(
        output,
        [
            {**base, "status": "backend_failure"},
            {**base, "status": "complete"},
        ],
    )
    assert not unresolved_backend_failure(output)


def _release_gate_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    dataset = tmp_path / "evidencegraph20k.jsonl"
    tasks = tmp_path / "tasks.jsonl"
    source_key = tmp_path / "source_key.jsonl"
    claim_ratings = tmp_path / "claim_ratings.csv"
    source_ratings = tmp_path / "source_ratings.csv"
    protocol = tmp_path / "AUDIT_PROTOCOL.md"
    for path in (dataset, tasks, source_key, claim_ratings, source_ratings, protocol):
        path.write_text(f"fixture:{path.name}\n", encoding="utf-8")

    audit_path = tmp_path / "audit_summary.json"
    audit = {
        "status": "complete",
        "protocol": "two-independent-raters-v2-blinded-aliases",
        "target_kappa": 0.8,
        "target_met": True,
        "metrics": {
            "n_claims": 3_000,
            "n_claim_ratings": 6_000,
            "rater_count": 2,
            "verdict_kappa": 0.85,
            "family_kappa": 0.82,
        },
        "inputs": {
            name: source_descriptor(path, tmp_path)
            for name, path in (
                ("dataset", dataset),
                ("tasks", tasks),
                ("source_key", source_key),
                ("claim_ratings", claim_ratings),
                ("source_ratings", source_ratings),
                ("audit_protocol", protocol),
            )
        },
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    registry_path = tmp_path / "dataset_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "display_id": "dataset",
                "registry": {"caption": "measured"},
                "sources": [
                    source_descriptor(dataset, tmp_path),
                    source_descriptor(audit_path, tmp_path),
                ],
            }
        ),
        encoding="utf-8",
    )

    release_root = tmp_path / "evidencegraph20k-v1"
    copied = {
        "data/evidencegraph20k.jsonl": dataset,
        "metadata/dataset_registry.json": registry_path,
        "audit/audit_summary.json": audit_path,
    }
    for relative, source in copied.items():
        destination = release_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    release_path = release_root / "release_manifest.json"
    release = {
        "schema_version": 1,
        "protocol": "evidencegraph20k-audited-release-v1",
        "status": "complete",
        "release_name": "evidencegraph20k-v1",
        "validation": {"episodes": 20_000, "test_claims": 3_000},
        "inputs": [
            source_descriptor(dataset, tmp_path),
            source_descriptor(audit_path, tmp_path),
            source_descriptor(registry_path, tmp_path),
        ],
        "outputs": [
            source_descriptor(release_root / relative, release_root)
            for relative in copied
        ],
    }
    release_path.write_text(json.dumps(release), encoding="utf-8")
    return audit_path, registry_path, release_path, release_root


def test_paid_transfer_requires_checksum_bound_human_audited_release(
    tmp_path: Path,
) -> None:
    audit, registry, release, release_root = _release_gate_fixture(tmp_path)

    result = verify_dataset_release_gate(
        audit, registry, release, project_root=tmp_path
    )

    assert result["n_claims"] == 3_000
    assert result["rater_count"] == 2
    assert result["release_manifest"]["sha256"]

    (release_root / "audit/audit_summary.json").write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="descriptor mismatch"):
        verify_dataset_release_gate(audit, registry, release, project_root=tmp_path)


def test_paid_transfer_rejects_below_target_human_audit(tmp_path: Path) -> None:
    audit, registry, release, _release_root = _release_gate_fixture(tmp_path)
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["target_met"] = False
    audit.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="complete genuine audit"):
        verify_dataset_release_gate(audit, registry, release, project_root=tmp_path)
