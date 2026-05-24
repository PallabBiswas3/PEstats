from __future__ import annotations
import numpy as np

class OutlierMixin:
    """
    Methods: remove_outliers, winsorize
    """

    def remove_outliers(
        self,
        method: str = "zscore",
        threshold: float = 3.0,
        fill: str = "interpolate"
    ) -> "OutlierMixin":
        """
        Detect and handle outliers.

        Parameters
        ----------
        method    : 'zscore'  — flag points beyond threshold standard deviations
                    'iqr'     — flag points outside 1.5×IQR (Tukey fences)
                    'mad'     — median absolute deviation (robust to spikes)
        threshold : multiplier for zscore/mad methods (default 3.0)
        fill      : what to do with flagged points
                    'interpolate' — replace with linear interpolation
                    'median'      — replace with rolling median
                    'nan'         — just mark as NaN (for inspection)
        """
        x = self.data.copy()
        mask = self._detect_outliers(x, method, threshold)

        if not mask.any():
            self._record(f"remove_outliers(method={method}) — none found")
            return self

        if fill == "interpolate":
            indices = np.arange(len(x))
            x[mask] = np.interp(indices[mask], indices[~mask], x[~mask])
        elif fill == "median":
            x[mask] = np.median(x[~mask])
        elif fill == "nan":
            x[mask] = np.nan
        else:
            raise ValueError(f"Unknown fill '{fill}'")

        n = mask.sum()
        self.data = x
        self._record(f"remove_outliers(method={method}, n_removed={n})")
        return self

    def _detect_outliers(
        self, x: np.ndarray, method: str, threshold: float
    ) -> np.ndarray:
        if method == "zscore":
            scores = np.abs((x - x.mean()) / (x.std() + 1e-12))
            return scores > threshold

        elif method == "iqr":
            q1, q3 = np.percentile(x, [25, 75])
            iqr = q3 - q1
            return (x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)

        elif method == "mad":
            median = np.median(x)
            mad = np.median(np.abs(x - median))
            scores = np.abs(x - median) / (mad * 1.4826 + 1e-12)
            return scores > threshold

        else:
            raise ValueError(f"Unknown method '{method}'. Choose: zscore, iqr, mad")

    def winsorize(self, limits: tuple = (0.05, 0.05)) -> "OutlierMixin":
        """
        Clip signal at given percentile limits instead of removing.
        E.g. limits=(0.05, 0.05) clips bottom 5% and top 5%.
        Less aggressive than remove_outliers — just caps extremes.
        """
        lower_pct = limits[0] * 100
        upper_pct = (1 - limits[1]) * 100
        lo = np.percentile(self.data, lower_pct)
        hi = np.percentile(self.data, upper_pct)
        self.data = np.clip(self.data, lo, hi)
        self._record(f"winsorize(limits={limits})")
        return self