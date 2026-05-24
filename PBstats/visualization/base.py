from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def apply_style() -> None:
    """
    Apply consistent pallabstats style to all matplotlib figures.
    Call once at the top of any plotting function.
    """
    mpl.rcParams.update({
        "figure.facecolor":     "white",
        "axes.facecolor":       "#f8f8f8",
        "axes.grid":            True,
        "grid.color":           "white",
        "grid.linewidth":       1.0,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.spines.left":     False,
        "axes.spines.bottom":   False,
        "axes.labelcolor":      "#333333",
        "axes.titlesize":       13,
        "axes.labelsize":       11,
        "xtick.color":          "#555555",
        "ytick.color":          "#555555",
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        "lines.linewidth":      1.6,
        "font.family":          "DejaVu Sans",
        "figure.dpi":           120,
    })

# Consistent color palette across all plots
COLORS = {
    "primary":   "#5B4FCF",   # purple  — main signal
    "secondary": "#1D9E75",   # teal    — comparison/envelope
    "accent":    "#D85A30",   # coral   — peaks/markers
    "neutral":   "#888780",   # gray    — reference lines
    "warning":   "#BA7517",   # amber   — thresholds
}

def save_or_show(fig: plt.Figure, save_path: str | None) -> None:
    """Save figure if path given, otherwise show it."""
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figure saved → {save_path}")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()