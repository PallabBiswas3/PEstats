"""Small shared helpers for the mixins."""
from __future__ import annotations

import sys
from typing import Callable

import numpy as np

_ASCII_FALLBACK = str.maketrans({
    "─": "-", "█": "#", "░": ".", "—": "-", "→": "->", "✖": "x", "▲": "!",
    "•": "*", "²": "^2", "×": "x", "σ": "sigma", "≥": ">=", "Δ": "d", "≈": "~",
})


def safe_print(*args, **kwargs) -> None:
    """print() that degrades to ASCII instead of crashing on consoles whose
    encoding can't show box-drawing characters (e.g. Windows cp1252)."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        text = kwargs.pop("sep", " ").join(str(a) for a in args).translate(_ASCII_FALLBACK)
        print(text.encode(enc, errors="replace").decode(enc), **kwargs)


def require_1d(x: np.ndarray, method: str) -> np.ndarray:
    """Raise a helpful error if ``x`` is multichannel."""
    if x.ndim != 1:
        raise ValueError(
            f"{method}() works on a single channel, but data has shape {x.shape}. "
            f"Select one channel first, e.g. d.channel(0).{method}()"
        )
    return x


def per_channel(fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
    """
    Apply a 1D -> 1D function to every channel of ``x``.

    Data convention: 1D arrays are a single channel; 2D arrays are
    ``(n_samples, n_channels)`` — time runs down the rows.
    """
    if x.ndim == 1:
        return np.asarray(fn(x), dtype=float)
    cols = [np.asarray(fn(x[:, c]), dtype=float) for c in range(x.shape[1])]
    return np.column_stack(cols)


def check_length(x: np.ndarray, min_len: int, method: str) -> None:
    n = x.shape[0]
    if n < min_len:
        raise ValueError(
            f"{method}() needs at least {min_len} samples, but the signal has {n}. "
            f"Use a lower filter order / fewer taps, or a longer recording."
        )


def interp_nans(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs in a 1D array (edges held constant)."""
    x = np.asarray(x, dtype=float).copy()
    mask = np.isnan(x)
    if mask.all():
        raise ValueError("Cannot interpolate: channel is entirely NaN.")
    if mask.any():
        idx = np.arange(len(x))
        x[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return x
