from __future__ import annotations
import numpy as np

class NormalizationMixin:
    """
    Methods: normalize, standardize, minmax, robust_scale
    """

    def normalize(self, method: str = "minmax") -> "NormalizationMixin":
        """
        Convenience wrapper — calls the right method by name.

        Parameters
        ----------
        method : 'minmax'  — scale to [0, 1]
                 'zscore'  — zero mean, unit variance
                 'robust'  — median and IQR based (outlier-resistant)
                 'l2'      — divide by L2 norm (unit vector)
        """
        dispatch = {
            "minmax": self.minmax,
            "zscore": self.standardize,
            "robust": self.robust_scale,
            "l2":     self._l2_norm,
        }
        if method not in dispatch:
            raise ValueError(f"Unknown method '{method}'. "
                             f"Choose: {list(dispatch)}")
        return dispatch[method]()

    def minmax(self) -> "NormalizationMixin":
        """Scale to [0, 1]."""
        x = self.data
        xmin, xmax = x.min(), x.max()
        if xmax == xmin:
            self.data = np.zeros_like(x)
        else:
            self.data = (x - xmin) / (xmax - xmin)
        self._record("minmax()")
        return self

    def standardize(self) -> "NormalizationMixin":
        """Zero mean, unit variance (z-score normalization)."""
        x = self.data
        std = x.std()
        if std == 0:
            self.data = np.zeros_like(x)
        else:
            self.data = (x - x.mean()) / std
        self._record("standardize()")
        return self

    def robust_scale(self) -> "NormalizationMixin":
        """
        Scale using median and IQR instead of mean and std.
        Much better for biomedical data with spikes or outliers.
        """
        x = self.data
        median = np.median(x)
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            self.data = np.zeros_like(x)
        else:
            self.data = (x - median) / iqr
        self._record("robust_scale()")
        return self

    def _l2_norm(self) -> "NormalizationMixin":
        """Divide by L2 norm — makes the signal a unit vector."""
        norm = np.linalg.norm(self.data)
        if norm == 0:
            self.data = np.zeros_like(self.data)
        else:
            self.data = self.data / norm
        self._record("l2_norm()")
        return self