from __future__ import annotations

import functools
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.labelcolor": "#333333",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.6,
    "font.family": "DejaVu Sans",
    "figure.dpi": 120,
}

# Consistent palette across all plots
COLORS = {
    "primary": "#5B4FCF",    # purple — main signal
    "secondary": "#1D9E75",  # teal   — comparison / envelope
    "accent": "#D85A30",     # coral  — peaks / markers
    "neutral": "#888780",    # gray   — reference lines
    "warning": "#BA7517",    # amber  — thresholds
}


@contextmanager
def style():
    """Apply the PBstats look *only* inside this block — the user's global
    matplotlib settings are left untouched."""
    with mpl.rc_context(STYLE):
        yield


def styled(fn):
    """Decorator: run a plotting function inside ``style()``."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with style():
            return fn(*args, **kwargs)
    return wrapper


def apply_style() -> None:
    """Apply the PBstats style globally (opt-in; plots don't need this)."""
    mpl.rcParams.update(STYLE)


def save_or_show(fig: plt.Figure, save_path: str | None, show: bool = True) -> None:
    """Save the figure if a path is given (and close it), else show it."""
    import warnings
    with warnings.catch_warnings():
        # gridspec/table layouts can't always be tightened; that's harmless
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout()
        except Exception:
            pass
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    elif show:
        plt.show()
