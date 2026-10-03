"""Consistent, accessible matplotlib styling for the notebook figures."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Colour-blind-validated categorical order; identity is always also given by labels.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def set_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "font.size": 10,
        "lines.linewidth": 2, "figure.dpi": 110,
    })


def pred_vs_true(ax, y, p, title: str) -> None:
    """Jittered scatter of predictions against the half-point ground-truth labels."""
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(y))
    ax.scatter(y + jitter, p, s=14, alpha=0.55, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    lims = [min(y.min(), p.min()) - 0.2, max(y.max(), p.max()) + 0.2]
    ax.plot(lims, lims, color=INK_MUTED, linewidth=1, linestyle="--", label="perfect prediction")
    ax.set(xlim=lims, ylim=lims, xlabel="True MOS grammar score", ylabel="Predicted score", title=title)
    ax.legend(frameon=False, loc="upper left")


def hbar(ax, labels, values, title: str, xlabel: str, color: str = BLUE) -> None:
    """Horizontal bars sorted by value, each labelled with its number."""
    order = np.argsort(values)
    vals = np.asarray(values)[order]
    colors = color if (vals >= 0).all() else np.where(vals >= 0, BLUE, ORANGE)  # diverging: sign gets its own hue
    ax.barh(np.asarray(labels)[order], vals, color=colors, height=0.65)
    ax.margins(x=0.15)  # room for the value labels at both ends
    for i, v in enumerate(vals):
        ax.text(v + (0.01 if v >= 0 else -0.01) * max(abs(vals).max(), 1e-9), i, f"{v:.2f}",
                va="center", ha="left" if v >= 0 else "right", color=INK_MUTED, fontsize=8)
    ax.axvline(0, color=INK_MUTED, linewidth=0.8)
    ax.set(title=title, xlabel=xlabel)
    ax.grid(axis="y", visible=False)
