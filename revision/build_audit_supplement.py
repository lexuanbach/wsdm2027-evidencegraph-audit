"""Generate supplementary tables directly from the checked retrospective grid."""
from pathlib import Path
import json
R = Path(__file__).resolve().parents[1]
d = json.loads((R / "revision/threshold-sensitivity.json").read_text())
tables = []
for kind, title, columns in [
    ("lineage", "Pair decisions at fixed lineage thresholds", [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("false_merges", "False links"), ("missed_links", "Missed links")]),
    ("lineage", "Partitions after transitive closure", [("mean_ari", "Mean ARI"), ("false_merged_pairs", "False merged pairs"), ("false_split_pairs", "False split pairs")]),
    ("text_reuse", "Pair decisions at fixed text-reuse thresholds", [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("false_merges", "False links"), ("missed_links", "Missed links")])]:
    lines = [r"\begin{center}\small", r"\begin{tabular}{ll" + "r" * len(columns) + "}", r"\toprule", "Scope & Threshold & " + " & ".join(c[1] for c in columns) + r" \\", r"\midrule"]
    for scope in ("natural", "constructed"):
        for threshold, metrics in d["scopes"][scope][kind].items():
            values = [str(metrics[key]) if isinstance(metrics[key], int) else f"{metrics[key]:.3f}" for key, _ in columns]
            lines.append(scope.capitalize() + " & " + threshold + " & " + " & ".join(values) + r" \\")
        if scope == "natural": lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    tables.append("\\subsection*{" + title + "}\n" + "\n".join(lines))
(R / "manuscript/generated/audit_sensitivity_tables.tex").write_text("\n\n".join(tables) + "\n")
print("Generated three supplementary tables from the stored grid")
