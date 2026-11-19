from pathlib import Path

import pytest

from evidencegraph.provenance import validate_file_preflight


def test_wikidata_preflight_rejects_partial_archive(tmp_path: Path) -> None:
    partial = tmp_path / "wikidata-20170220-all.json.bz2"
    partial.write_bytes(b"partial")

    with pytest.raises(
        ValueError,
        match=r"wikidata_20170220 has 7 bytes; expected 7166563790",
    ):
        validate_file_preflight("wikidata_20170220", partial)


def test_preflight_rejects_absent_input(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError, match="missing.jsonl"):
        validate_file_preflight("unregistered_input", missing)
