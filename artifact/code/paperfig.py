"""Shared figure style for the EvidenceGraph manuscript (WSDM '27).

Categorical palette is colorblind-validated (CVD dE >= 8 adjacent pairs on a
white surface). Color assignment is fixed per method across every figure in
the paper; series identity is additionally carried by marker shape and line
style so figures survive grayscale printing.
"""
import matplotlib as mpl

BLUE = "#2a78d6"     # EvidenceGraph (ours)
ORANGE = "#eb6834"   # ReAct
AQUA = "#1baf7a"     # reserved auxiliary series
YELLOW = "#eda100"   # Adaptive-RAG
MAGENTA = "#e87ba4"  # fixed-budget RAG
VIOLET = "#4a3aa7"   # RL cost-penalty
GRAY = "#6b6a66"     # oracle / reference lines


def apply_style():
    mpl.rcParams.update({
        "font.size": 7.5,
        "font.family": "sans-serif",
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.4,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.30,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


# Camera-ready style: text set by LaTeX with the paper's Libertine fonts, a
# muted palette shared with Figure 1, light grids, and no top/right spines.
INK, MUTE, GRIDC = "#2B3441", "#8A919C", "#E3E6EA"
NAVY, TEAL, RUST, OCHRE, PLUM = "#2F4B66", "#2F7470", "#B4583C", "#A8781F", "#6A4C6B"


def apply_camera_ready_style():
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[tt=false,type1=true]{libertine}"
                               r"\usepackage[libertine]{newtxmath}",
        "font.family": "serif",
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
        "axes.edgecolor": "#9AA1AB", "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": "#5A616C", "ytick.color": "#5A616C",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "xtick.major.pad": 1.6, "ytick.major.pad": 1.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.5,
        "grid.alpha": 1.0,
        "axes.axisbelow": True, "axes.titlepad": 3, "axes.labelpad": 1.5,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
    })

