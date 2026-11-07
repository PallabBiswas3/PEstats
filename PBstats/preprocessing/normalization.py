from __future__ import annotations

import numpy as np

from PBstats.core.steps import step


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den, giving 0 wherever den == 0 (constant channels)."""
    den = np.asarray(den, dtype=float)
    out = np.zeros_like(num, dtype=float)
    ok = np.broadcast_to(den != 0, num.shape)
    np.divide(num, np.broadcast_to(den, num.shape), out=out, where=ok)
    return out


class NormalizationMixin:
    """normalize, standardize, minmax, robust_scale — all per channel."""

    @step
    def normalize(self, method: str = "minmax") -> "NormalizationMixin":
        """
        Dispatch by name: 'minmax' → [0, 1], 'zscore' → zero mean / unit
        variance, 'robust' → median / IQR, 'l2' → unit L2 norm.
        """
        dispatch = {
            "minmax": self.minmax,
            "zscore": self.standardize,
            "robust": self.robust_scale,
            "l2": self._l2_norm,
        }
        if method not in dispatch:
            raise ValueError(f"Unknown method '{method}'. Choose: {list(dispatch)}")
        return dispatch[method]()

    @step
    def minmax(self) -> "NormalizationMixin":
        """Scale each channel to [0, 1]. Constant channels become 0."""
        x = self.data
        lo, hi = np.nanmin(x, axis=0), np.nanmax(x, axis=0)
        self.data = _safe_divide(x - lo, hi - lo)
        self._record("minmax()")
        return self

    @step
    def standardize(self) -> "NormalizationMixin":
        """Zero mean, unit variance per channel (z-score)."""
        x = self.data
        self.data = _safe_divide(x - np.nanmean(x, axis=0), np.nanstd(x, axis=0))
        self._record("standardize()")
        return self

    @step
    def robust_scale(self) -> "NormalizationMixin":
        """
        Centre on the median, scale by the IQR. Much better than z-score for
        biomedical data with spikes or outliers.
        """
        x = self.data
        q25, med, q75 = np.nanpercentile(x, [25, 50, 75], axis=0)
        self.data = _safe_divide(x - med, q75 - q25)
        self._record("robust_scale()")
        return self

    def _l2_norm(self) -> "NormalizationMixin":
        """Divide each channel by its L2 norm."""
        self.data = _safe_divide(self.data, np.linalg.norm(self.data, axis=0))
        self._record("l2_norm()")
        return self
