from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter

from PBstats.core.steps import step
from PBstats.core.utils import per_channel

_MAD_TO_STD = 1.4826  # makes MAD a consistent estimator of std for Gaussian data


def rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window) | 1)  # odd, >= 3
    return median_filter(x, size=window, mode="nearest")


def detect_outliers(
    x: np.ndarray, method: str = "zscore", threshold: float = 3.0, window: int = 11,
) -> np.ndarray:
    """Boolean mask of outlying samples in a 1D array (NaNs are never flagged)."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if method == "zscore":
        mu, sd = np.nanmean(x), np.nanstd(x)
        scores = np.abs(x - mu) / (sd + 1e-12)
    elif method == "iqr":
        q1, q3 = np.nanpercentile(x, [25, 75])
        iqr = q3 - q1
        return finite & ((x < q1 - threshold * iqr) | (x > q3 + threshold * iqr))
    elif method == "mad":
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        scores = np.abs(x - med) / (_MAD_TO_STD * mad + 1e-12)
    elif method == "hampel":
        # Local (rolling) median/MAD — follows trends and slow oscillations,
        # so it catches spikes that a global threshold would miss.
        xf = np.where(finite, x, np.nanmedian(x))
        med = rolling_median(xf, window)
        resid = xf - med
        local = _MAD_TO_STD * rolling_median(np.abs(resid), window)
        # The local scale collapses to ~0 on smooth stretches, which would flag
        # harmless curvature. Floor it with the robust *global* residual scale,
        # itself floored at 2% of the signal's robust spread.
        spread = _MAD_TO_STD * np.median(np.abs(xf - np.median(xf)))
        global_scale = _MAD_TO_STD * np.median(np.abs(resid - np.median(resid)))
        floor = max(global_scale, 0.02 * spread, 1e-12)
        scores = np.abs(x - med) / np.maximum(local, floor)
    elif method == "spike":
        # Impulse-artefact detector. A sample is a spike when it departs from
        # the short rolling median by much more than BOTH the noise level and
        # the local steepness of the underlying signal. The slope term is what
        # keeps sharp-but-real waveform features (ECG R-peaks, the crest of a
        # fast oscillation) from being mistaken for artefacts.
        xf = np.where(finite, x, np.nanmedian(x))
        med = rolling_median(xf, window)
        # Noise from first differences (robust, ≈ σ for white noise). The
        # median-filter residual can't be used: it is exactly 0 for most
        # samples, so its MAD collapses to 0 and every wiggle looks huge.
        d = np.diff(xf)
        noise = _MAD_TO_STD * np.median(np.abs(d - np.median(d))) / np.sqrt(2) if len(d) else 0.0
        spread = _MAD_TO_STD * np.median(np.abs(xf - np.median(xf)))
        slope = maximum_filter1d(np.abs(np.gradient(med)), size=max(3, int(window)), mode="nearest")
        scale = np.maximum(max(noise, 0.02 * spread, 1e-12), slope)
        scores = np.abs(x - med) / scale
    else:
        raise ValueError(f"Unknown method '{method}'. Choose: zscore, iqr, mad, hampel, spike")
    return finite & (scores > threshold)


class OutlierMixin:
    """remove_outliers, winsorize — per channel."""

    @step
    def remove_outliers(
        self,
        method: str = "zscore",
        threshold: float | None = None,
        fill: str = "interpolate",
        window: int = 11,
    ) -> "OutlierMixin":
        """
        Detect and replace outliers.

        Parameters
        ----------
        method    : 'zscore' — beyond threshold standard deviations (default 3)
                    'iqr'    — outside Tukey fences, threshold × IQR (default 1.5)
                    'mad'    — robust global median absolute deviation (default 3)
                    'hampel' — robust *rolling* MAD; best for spikes riding on
                               a signal with trends/oscillations (default 3)
                    'spike'  — impulse artefacts only: deviation from a short
                               rolling median vs. noise AND local slope, so
                               sharp real peaks (ECG R-waves) are kept.
                               Use a high threshold, e.g. 10.
        threshold : sensitivity; None uses the per-method default above
        fill      : 'interpolate' — linear interpolation from good neighbours
                    'median'      — rolling median of length ``window``
                    'nan'         — mark as NaN for inspection
        window    : samples in the rolling window (hampel / median fill)
        """
        if threshold is None:
            threshold = 1.5 if method == "iqr" else 3.0
        if fill not in ("interpolate", "median", "nan"):
            raise ValueError(f"Unknown fill '{fill}'. Choose: interpolate, median, nan")

        counter = {"n": 0}

        def _clean(c: np.ndarray) -> np.ndarray:
            c = c.astype(float).copy()
            mask = detect_outliers(c, method, threshold, window)
            counter["n"] += int(mask.sum())
            if not mask.any():
                return c
            if fill == "nan":
                c[mask] = np.nan
            elif fill == "median":
                good = np.where(mask, np.nan, c)
                c[mask] = rolling_median(
                    np.where(np.isnan(good), np.nanmedian(good), good), window
                )[mask]
            else:
                good = ~mask & np.isfinite(c)
                if not good.any():
                    return c
                idx = np.arange(len(c))
                c[mask] = np.interp(idx[mask], idx[good], c[good])
            return c

        self.data = per_channel(_clean, self.data)
        self._record(f"remove_outliers(method={method}, n_removed={counter['n']})")
        return self

    @step
    def winsorize(self, limits: tuple = (0.05, 0.05)) -> "OutlierMixin":
        """
        Cap each channel at percentile limits instead of removing points.
        limits=(0.05, 0.05) caps the bottom and top 5%.
        """
        lo = np.nanpercentile(self.data, limits[0] * 100, axis=0)
        hi = np.nanpercentile(self.data, (1 - limits[1]) * 100, axis=0)
        self.data = np.clip(self.data, lo, hi)
        self._record(f"winsorize(limits={tuple(limits)})")
        return self
