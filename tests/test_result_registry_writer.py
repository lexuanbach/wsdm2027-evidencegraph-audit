from __future__ import annotations

from pathlib import Path

import pytest

from evidencegraph.result_registry import (
    complete_registry_envelope,
    write_complete_registry,
)


def test_complete_registry_records_relative_hash_and_jsonl_rows(tmp_path: Path) -> None:
    source = tmp_path / "run.jsonl"
    source.write_text("{}\n{}\n", encoding="utf-8")
    envelope = complete_registry_envelope(
        "main", {"rows": [1]}, [source], project_root=tmp_path
    )
    assert envelope["sources"][0]["path"] == "run.jsonl"
    assert envelope["sources"][0]["rows"] == 2
    assert len(envelope["sources"][0]["sha256"]) == 64


def test_registry_writer_refuses_changed_existing_measurement(tmp_path: Path) -> None:
    source = tmp_path / "run.json"
    source.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "registries"
    write_complete_registry(
        "main", {"rows": [1]}, [source], output_dir, project_root=tmp_path
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_complete_registry(
            "main", {"rows": [2]}, [source], output_dir, project_root=tmp_path
        )
