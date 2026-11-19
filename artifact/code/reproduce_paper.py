"""Rebuild every paper and supplement result that is derived from stored outputs.

Usage (from the repository root)::

    python artifact/code/reproduce_paper.py            # rebuild and verify
    python artifact/code/reproduce_paper.py --no-tex   # skip LaTeX text in the figure

The script makes no model or network calls. It reads the stored summaries in
``artifact/runs/`` and ``annotation/score/``, writes regenerated outputs to
``outputs/``, and checks them against the camera-ready values:

* Tables 1-6 of the paper, plus a numeric companion of Figure 3, are rebuilt
  as table fragments and compared byte for byte with the camera-ready
  versions, whose SHA-256 digests are stored in ``expected/paper_tables.sha256``.
* Figure 3 (acquisition on natural lineage) is redrawn from its 18 stored
  estimates, which are also written to ``outputs/figures/fig_natural_values.csv``.
* The supplement's three threshold tables (fixed lineage and text-reuse
  thresholds) are recomputed from the deidentified audit labels and compared
  with the stored grid.
* The supplement's two historical covariance tables are recomputed from the
  three saved prediction files and compared with the stored values.

It exits with status 1 if any rebuilt value differs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "artifact/code"
RUNS = ROOT / "artifact/runs"
sys.path.insert(0, str(CODE))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, tol: float = 1e-9) -> bool:
    """Recursive equality with a small float tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    return a == b


def rebuild_tables(out: Path) -> list[str]:
    import build_subset_tables as tables

    tables.PROJECT_ROOT = ROOT
    tables.SUB = RUNS / "subset_edge_episode"
    tables.TABLE_DIR = out / "tables"
    argv, sys.argv = sys.argv, ["build_subset_tables.py"]
    try:
        tables.main()
    finally:
        sys.argv = argv
    failures = []
    expected = {}
    for line in (ROOT / "expected/paper_tables.sha256").read_text().splitlines():
        digest, name = line.split()
        expected[name] = digest
    for name, digest in sorted(expected.items()):
        path = out / "tables" / name
        ok = path.is_file() and sha256(path) == digest
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
        if not ok:
            failures.append(f"tables/{name}")
    return failures


def rebuild_natural_figure(out: Path, use_tex: bool) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import paperfig
    import make_natural_figure as fig

    fig.SUMMARY = RUNS / "natural_lineage_acquisition/summary.json"
    fig.FIGURE_DIR = out / "figures"
    if not use_tex:
        from matplotlib.axes import Axes

        style = paperfig.apply_camera_ready_style

        def plain_style() -> None:
            style()
            matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif"})

        paperfig.apply_camera_ready_style = plain_style
        fig.POLICIES = tuple(
            (key, label.replace(r"\textbf{EvidenceGraph}", "EvidenceGraph"))
            for key, label in fig.POLICIES
        )
        set_xlabel = Axes.set_xlabel
        Axes.set_xlabel = lambda self, text, *a, **k: set_xlabel(
            self, text.replace("\\ ", " "), *a, **k
        )
    fig.main()

    summary = json.loads(fig.SUMMARY.read_text(encoding="utf-8"))
    rows = []
    for key, label, *_ in fig.DATASETS:
        block = summary["datasets"][key]
        for policy, _ in fig.POLICIES:
            d = block["item_recall"][policy]["difference_vs_reference"]
            rows.append([label, policy, f"{100 * d['difference']:.2f}",
                         f"{100 * d['low']:.2f}", f"{100 * d['high']:.2f}"])
    with (out / "figures/fig_natural_values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "policy", "recall_change_points", "ci_low", "ci_high"])
        writer.writerows(rows)
    ok = len(rows) == 18
    print(f"  {'ok  ' if ok else 'FAIL'} figures/fig_natural.pdf and fig_natural_values.csv ({len(rows)} estimates)")
    return [] if ok else ["figures/fig_natural"]


def rebuild_threshold_grid(out: Path) -> list[str]:
    from threshold_sensitivity import analyze

    inputs = RUNS / "lineage_threshold/audit_inputs.json"
    stored = RUNS / "lineage_threshold/threshold_sensitivity.json"
    result = analyze(json.loads(inputs.read_text()))
    (out / "supplement").mkdir(parents=True, exist_ok=True)
    (out / "supplement/threshold_sensitivity.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = []
    for kind, title, cols in (
        ("lineage", "Pair decisions at fixed lineage thresholds",
         ["precision", "recall", "f1", "false_merges", "missed_links"]),
        ("lineage", "Partitions after transitive closure",
         ["mean_ari", "false_merged_pairs", "false_split_pairs"]),
        ("text_reuse", "Pair decisions at fixed text-reuse thresholds",
         ["precision", "recall", "f1", "false_merges", "missed_links"]),
    ):
        lines += [f"### {title}", "", "| Scope | Threshold | " + " | ".join(cols) + " |",
                  "|---|---|" + "---|" * len(cols)]
        for scope in ("natural", "constructed"):
            for threshold, m in result["scopes"][scope][kind].items():
                cells = [str(m[c]) if isinstance(m[c], int) else f"{m[c]:.3f}" for c in cols]
                lines.append(f"| {scope} | {threshold} | " + " | ".join(cells) + " |")
        lines.append("")
    (out / "supplement/threshold_tables.md").write_text("\n".join(lines))
    ok = close(result, json.loads(stored.read_text()))
    print(f"  {'ok  ' if ok else 'FAIL'} supplement threshold tables (natural F1 .568, ARI .401 at 0.5)")
    return [] if ok else ["supplement/threshold_sensitivity"]


def rebuild_covariance(out: Path) -> list[str]:
    from saved_covariance_metrics import analyze

    directory = RUNS / "saved_covariance"
    result = analyze(directory)
    (out / "supplement").mkdir(parents=True, exist_ok=True)
    (out / "supplement/saved_covariance_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    names = {"ridge_covariance": "Covariance ridge", "ridge_precision": "Precision ridge",
             "uniform_mean_stance": "Uniform"}
    lines = ["### Historical covariance arms", "",
             "| Arm | Split | Acc. | Log loss | Brier | ECE | Mean n_eff |", "|---|---|---|---|---|---|---|"]
    for arm, label in names.items():
        for split in ("dev", "test"):
            m = result["metrics"][arm][split]["overall"]
            lines.append(f"| {label} | {split} | {m['accuracy']:.4f} | {m['log_loss']:.4f} | "
                         f"{m['brier']:.4f} | {m['ece']:.4f} | {m['mean_effective_evidence']:.2f} |")
    lines += ["", "### Ridge minus uniform on test, paired bootstrap (1,000 draws, seed 217)", "",
              "| Arm / test subset | Metric | Difference | 95% interval |", "|---|---|---|---|"]
    for arm in ("ridge_covariance", "ridge_precision"):
        for subset, values in result["paired_test"][arm].items():
            for metric in ("accuracy", "log_loss", "brier", "ece"):
                v = values[metric]
                lines.append(f"| {names[arm]} / {subset} | {metric} | {v['difference']:+.4f} | "
                             f"[{v['low']:+.4f}, {v['high']:+.4f}] |")
    (out / "supplement/covariance_tables.md").write_text("\n".join(lines) + "\n")
    ok = close(result, json.loads((directory / "saved-prediction-audit.json").read_text()))
    print(f"  {'ok  ' if ok else 'FAIL'} supplement covariance tables (359 dev / 371 test claims)")
    return [] if ok else ["supplement/saved_covariance"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    parser.add_argument("--no-tex", action="store_true",
                        help="draw the figure without a LaTeX installation")
    args = parser.parse_args()
    use_tex = not args.no_tex and shutil.which("latex") is not None
    if not args.no_tex and not use_tex:
        print("note: no 'latex' on PATH, drawing the figure with matplotlib text")
    args.out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    print("Paper tables (byte-identical to the camera-ready fragments):")
    failures += rebuild_tables(args.out)
    print("Figure 3:")
    failures += rebuild_natural_figure(args.out, use_tex)
    print("Supplement:")
    failures += rebuild_threshold_grid(args.out)
    failures += rebuild_covariance(args.out)
    if failures:
        print("MISMATCH: " + ", ".join(failures))
        sys.exit(1)
    print(f"All stored-output results reproduced. Outputs are in {args.out}.")


if __name__ == "__main__":
    main()
