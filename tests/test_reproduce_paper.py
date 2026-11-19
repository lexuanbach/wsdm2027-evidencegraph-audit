"""End-to-end check: every stored-output result in the paper and supplement rebuilds."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_reproduce_paper_rebuilds_all_tables_figure_and_supplement(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "artifact/code/reproduce_paper.py"), "--no-tex", "--out", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All stored-output results reproduced" in result.stdout
    expected = {line.split()[1] for line in (ROOT / "expected/paper_tables.sha256").read_text().splitlines()}
    assert expected == {path.name for path in (tmp_path / "tables").glob("*.tex")}
    assert (tmp_path / "figures/fig_natural.pdf").stat().st_size > 0
    grid = json.loads((tmp_path / "supplement/threshold_sensitivity.json").read_text())
    natural = grid["scopes"]["natural"]["lineage"]["0.5"]
    assert round(natural["f1"], 3) == 0.568 and round(natural["mean_ari"], 3) == 0.401


def test_saved_covariance_metrics_match_stored_values() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "artifact/code/saved_covariance_metrics.py"), "--verify"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
