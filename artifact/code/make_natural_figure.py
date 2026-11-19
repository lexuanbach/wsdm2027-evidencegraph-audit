"""Forest plot of acquisition on *observed* Web lineage (Figure 3).

Replaces the numeric matrix that used to be ``tab:natural``.  Every point and
every interval is read from
``artifact/runs/natural_lineage_acquisition/summary.json``:

    datasets.<ds>.item_recall.<policy>.difference_vs_reference.{difference,low,high}

for the six non-reference policies on each of the three datasets --- 18
estimates and 18 paired 95% claim-bootstrap intervals.  The absolute passive
next-rank recalls (``item_recall.passive_rank.mean``) are carried in the legend
so the denominators of the differences stay in the main text: a 1.3-point loss
on a base of 8.7 is not the same thing as a 1.6-point loss on a base of 66.8.

Nothing is smoothed, fitted or interpolated; the script only reshapes stored
numbers.  Regenerate with::

    PYTHONPATH=artifact/code python artifact/code/make_natural_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

import paperfig as P


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY = PROJECT_ROOT / "artifact/runs/natural_lineage_acquisition/summary.json"
FIGURE_DIR = PROJECT_ROOT / "manuscript/figures"
# fallback layout: code/, runs/ and manuscript/ are siblings
ART = Path(__file__).resolve().parents[1]
if not SUMMARY.exists():
    SUMMARY = ART / "runs/natural_lineage_acquisition/summary.json"
    FIGURE_DIR = ART / "manuscript/figures"

# top row first; the oracle is the ceiling, ours is the bottom row
POLICIES = (
    ("oracle_gold_ceiling", "oracle that sees gold"),
    ("mmr_text", "text diversity (MMR)"),
    ("neardup_diverse", "skip near-duplicates"),
    ("observed_origin_diverse", "one per recorded origin"),
    ("learned_hard_dependence", "learned lineage, hard"),
    ("learned_soft_find_indep", r"\textbf{EvidenceGraph}, soft"),
)
DATASETS = (
    ("averitec", "AVeriTeC", P.NAVY, "o", 0.24),
    ("fever", "FEVER", P.OCHRE, "s", 0.0),
    ("hover", "HoVer", P.PLUM, "^", -0.24),
)


def main() -> None:
    P.apply_camera_ready_style()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if summary.get("reference") != "passive_rank":
        raise ValueError("summary reference policy is not passive_rank")

    fig, ax = plt.subplots(figsize=(3.34, 1.66))
    n_points = 0
    for key, label, color, marker, offset in DATASETS:
        block = summary["datasets"][key]
        passive = 100.0 * block["item_recall"]["passive_rank"]["mean"]
        legend = f"{label}, $n$={block['n_cases']}, rank-order recall {passive:.1f}"
        first = True
        for row, (policy, _) in enumerate(POLICIES):
            diff = block["item_recall"][policy]["difference_vs_reference"]
            y = len(POLICIES) - 1 - row + offset
            lo, hi = 100.0 * diff["low"], 100.0 * diff["high"]
            point = 100.0 * diff["difference"]
            ax.plot([lo, hi], [y, y], color=color, lw=1.1,
                    solid_capstyle="round", zorder=3)
            ax.plot(
                [point], [y], color=color, marker=marker, ms=3.6, ls="none",
                mec="white", mew=0.5, zorder=4,
                label=legend if first else None,
            )
            first = False
            n_points += 1

    ax.axhspan(len(POLICIES) - 1.5, len(POLICIES) - 0.5, color="#EEF2F6",
               lw=0, zorder=0)              # the oracle row is a ceiling
    ax.axvline(0.0, color=P.INK, lw=0.6, ls=(0, (1.2, 1.2)), zorder=2)
    ax.set_yticks(range(len(POLICIES)))
    ax.set_yticklabels([label for _, label in reversed(POLICIES)])
    ax.set_ylim(-0.65, len(POLICIES) - 0.35)
    ax.set_xlabel("change in gold-evidence recall vs.\\ rank order (points)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-4.0, 9.4)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), ncol=1,
              fontsize=6.0, handletextpad=0.3, borderpad=0.1, labelspacing=0.25)
    fig.tight_layout(pad=0.15)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "fig_natural.pdf"
    fig.savefig(out)
    plt.close(fig)
    if n_points != len(POLICIES) * len(DATASETS):
        raise ValueError(f"expected 18 estimates, drew {n_points}")
    print(f"wrote {out} from {n_points} stored estimates and {n_points} intervals")


if __name__ == "__main__":
    main()
