from __future__ import annotations

import numpy as np

from PBstats.core.steps import step
from PBstats.core.utils import interp_nans, per_channel

_FILL_STRATEGIES = ("interpolate", "zero", "mean", "median", "ffill", "bfill", "drop")


def _ffill_1d(x: np.ndarray) -> np.ndarray:
    # Index of the last valid sample at or before each position. Leading NaNs
    # map to index 0, which is itself NaN, so they correctly stay NaN.
    idx = np.where(~np.isnan(x), np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    return x[idx]


def _bfill_1d(x: np.ndarray) -> np.ndarray:
    return _ffill_1d(x[::-1])[::-1]


def fill_missing_1d(x: np.ndarray, strategy: str) -> np.ndarray:
    """Fill NaNs in a single channel. ``drop`` is handled by the caller."""
    x = np.asarray(x, dtype=float)
    if not np.isnan(x).any():
        return x.copy()
    if strategy == "interpolate":
        return interp_nans(x)
    if strategy == "zero":
        return np.nan_to_num(x, nan=0.0)
    if strategy == "mean":
        return np.where(np.isnan(x), np.nanmean(x), x)
    if strategy == "median":
        return np.where(np.isnan(x), np.nanmedian(x), x)
    if strategy == "ffill":
        return _ffill_1d(x)
    if strategy == "bfill":
        return _bfill_1d(x)
    raise ValueError(f"Unknown strategy '{strategy}'. Choose: {', '.join(_FILL_STRATEGIES)}")


class CleaningMixin:
    """remove_missing, remove_dc, clip, crop — each returns self."""

    @step
    def remove_missing(self, strategy: str = "interpolate") -> "CleaningMixin":
        """
        Replace NaN values. Works per channel on multichannel data.

        Parameters
        ----------
        strategy : 'interpolate' — linear interpolation between neighbours
                   'zero' / 'mean' / 'median' — constant fill
                   'ffill' / 'bfill' — carry last / next valid value
                   'drop' — remove NaN samples (whole rows for 2D data).
                            Note: this breaks uniform sampling.
        """
        if strategy not in _FILL_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose: {', '.join(_FILL_STRATEGIES)}"
            )
        x = self.data.astype(float)
        n_missing = int(np.isnan(x).sum())

        if strategy == "drop":
            keep = ~np.isnan(x) if x.ndim == 1 else ~np.isnan(x).any(axis=1)
            x = x[keep]
        else:
            x = per_channel(lambda c: fill_missing_1d(c, strategy), x)

        self.data = x
        self._record(f"remove_missing(strategy={strategy}, n_missing={n_missing})")
        return self

    @step
    def remove_dc(self) -> "CleaningMixin":
        """
        Subtract the mean of each channel. Essential before spectral
        analysis — DC shows up as a huge 0 Hz spike otherwise.
        """
        self.data = self.data - np.nanmean(self.data, axis=0)
        self._record("remove_dc()")
        return self

    @step
    def clip(self, low: float, high: float) -> "CleaningMixin":
        """Hard-clip values to [low, high]."""
        if low > high:
            raise ValueError(f"low ({low}) must be <= high ({high})")
        self.data = np.clip(self.data, low, high)
        self._record(f"clip(low={low}, high={high})")
        return self

    @step
    def crop(self, start: float = 0.0, end: float | None = None) -> "CleaningMixin":
        """
        Keep only the segment [start, end). Units are seconds when ``fs`` is
        set, otherwise sample indices.
        """
        scale = self.fs if self.fs else 1.0
        n = self.data.shape[0]
        i0 = int(round(start * scale))
        i1 = n if end is None else int(round(end * scale))
        i0, i1 = max(i0, 0), min(i1, n)
        if i1 <= i0:
            raise ValueError(f"Empty crop: start={start}, end={end} (signal has {n} samples)")
        self.data = self.data[i0:i1].copy()
        # Keep attached events aligned with the new time origin.
        ann = getattr(self, "annotations", None)
        if ann is not None and len(ann) and self.fs:
            t0, t1 = i0 / self.fs, i1 / self.fs
            ann = ann[(ann["onset"] >= t0) & (ann["onset"] < t1)].copy()
            ann["onset"] -= t0
            self.annotations = ann.reset_index(drop=True)
        self._record(f"crop(start={start}, end={end})")
        return self
