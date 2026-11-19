"""Duplication-intervention figure from the frozen three-seed summaries.

Two backbones (Qwen2.5-3B-Instruct and Qwen2.5-1.5B), count-based RAG versus
known-provenance collapse, plus the dedup-equivalent reference: collapsing
exact copies reproduces the one-copy context, so any URL/near-duplicate
deduplicator sits at the k=1 fixed-RAG value by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import paperfig as P


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "manuscript/figures"
SUMMARIES = {
    "Qwen2.5-3B": PROJECT_ROOT / "artifact/runs/duplication/qwen3b-v6-three-seed-summary.json",
    "Qwen2.5-1.5B": PROJECT_ROOT / "artifact/runs/duplication/scale-qwen15b-three-seed-summary.json",
}
STYLE = {
    ("Qwen2.5-3B", "fixed_rag"): (P.MAGENTA, "^", "-", "Count-based RAG, 3B"),
    ("Qwen2.5-3B", "evidencegraph_oracle"): (P.BLUE, "o", "-", "Known-provenance collapse, 3B"),
    ("Qwen2.5-1.5B", "fixed_rag"): (P.ORANGE, "v", "--", "Count-based RAG, 1.5B"),
    ("Qwen2.5-1.5B", "evidencegraph_oracle"): (P.VIOLET, "s", "--", "Known-provenance collapse, 1.5B"),
}


def main() -> None:
    P.apply_style()
    data = {name: json.load(path.open(encoding="utf-8")) for name, path in SUMMARIES.items()}
    for name, summary in data.items():
        if summary["n_claims"] != 40 or summary["n_seeds"] != 3 or not summary["diagnostic"]["headline_holds"]:
            raise ValueError(f"duplication summary for {name} is not the frozen three-seed run")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    for name, summary in data.items():
        x = np.asarray(summary["copy_count"], dtype=float)
        for method, record in summary["methods"].items():
            color, marker, ls, label = STYLE[(name, method)]
            for ax, key in zip(axes, ("confidence", "false_acceptance")):
                y = np.asarray(record[f"{key}_mean"], dtype=float)
                ax.plot(x, y, color=color, marker=marker, ls=ls, label=label)
                ax.fill_between(x, record[f"{key}_low"], record[f"{key}_high"], color=color, alpha=0.12, linewidth=0)
        # dedup-equivalent reference: k=1 count-based value, flat.
        base = summary["methods"]["fixed_rag"]
        for ax, key in zip(axes, ("confidence", "false_acceptance")):
            ax.axhline(base[f"{key}_mean"][0], color=P.GRAY, lw=0.8, ls=":", zorder=0)
    for ax in axes:
        ax.set_xscale("log")
        x = np.asarray(data["Qwen2.5-3B"]["copy_count"], dtype=float)
        ax.set_xticks(x, [str(int(v)) for v in x])
        ax.set_xlabel("copies of one unsupported source (20 slots)")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y")
    axes[0].set_ylabel("mean P(false claim)")
    axes[0].set_title("Confidence response")
    axes[1].set_ylabel("false-claim acceptance")
    axes[1].set_title("Decision response")
    # One shared legend under both panels: the four series are identical in each,
    # so a per-axes legend both duplicates it and overlaps the curves it labels.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=5.6,
               frameon=False, bbox_to_anchor=(0.5, -0.02), columnspacing=1.4,
               handlelength=1.8)
    fig.tight_layout(w_pad=1.4, rect=(0, 0.10, 1, 1))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "fig_duplication.pdf")
    plt.close(fig)
    print("wrote", FIGURE_DIR / "fig_duplication.pdf")


if __name__ == "__main__":
    main()
