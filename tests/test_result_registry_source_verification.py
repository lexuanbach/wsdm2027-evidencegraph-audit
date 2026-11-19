from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidencegraph.result_registry import source_descriptor, verified_source_path


def test_verified_source_path_binds_jsonl_bytes_and_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"id": 1}) + "\n", encoding="utf-8")
    descriptor = source_descriptor(source, tmp_path)

    assert verified_source_path(descriptor, tmp_path) == source
    descriptor["rows"] = 2
    with pytest.raises(ValueError, match="rows"):
        verified_source_path(descriptor, tmp_path)
