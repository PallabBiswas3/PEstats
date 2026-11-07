from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from PBstats.core.steps import step
from PBstats.statistics.base import DescriptiveResult


class DescriptiveMixin:

    @step
    def describe(self) -> "DescriptiveMixin":
        """
        Full descriptive statistics of ``self.data`` (NaNs excluded),
        stored in ``self.descriptive_result``. Non-destructive.

        Adds signal metrics plain stats libraries omit: RMS, MAD and a
        heuristic SNR (signal RMS vs. noise estimated from first differences —
        reasonable for oversampled signals, optimistic for white-ish ones).
        """
        x = self.data.flatten().astype(float)
        x = x[~np.isnan(x)]
        if len(x) < 2:
            raise ValueError("describe() needs at least 2 non-NaN samples")

        rms = float(np.sqrt(np.mean(x ** 2)))
        noise_est = float(np.std(np.diff(x), ddof=1)) / np.sqrt(2)
        snr_db = 20 * np.log10(rms / noise_est) if noise_est > 0 else float("inf")
        q25, med, q75 = np.percentile(x, [25, 50, 75])

        self.descriptive_result = DescriptiveResult(
            mean=float(np.mean(x)),
            median=float(med),
            std=float(np.std(x, ddof=1)),
            variance=float(np.var(x, ddof=1)),
            skewness=float(sp_stats.skew(x)),
            kurtosis=float(sp_stats.kurtosis(x)),
            minimum=float(x.min()),
            maximum=float(x.max()),
            q25=float(q25),
            q75=float(q75),
            iqr=float(q75 - q25),
            rms=rms,
            snr=float(snr_db),
            n=len(x),
            label=self.label,
            mad=float(np.median(np.abs(x - med))),
            values=x,
        )
        self._record("describe()")
        return self

    def peak_to_peak(self) -> float:
        """Peak-to-peak amplitude. Non-destructive."""
        return float(np.nanmax(self.data) - np.nanmin(self.data))

    def energy(self) -> float:
        """Total signal energy (sum of squares). Non-destructive."""
        return float(np.nansum(self.data ** 2))

    def zero_crossing_rate(self) -> float:
        """
        Fraction of consecutive sample pairs that cross zero (computed on the
        mean-removed signal). Useful for voiced/unvoiced audio, EMG tremor.
        """
        x = self.data.flatten()
        x = x - np.nanmean(x)
        s = np.signbit(x)
        return float(np.count_nonzero(s[1:] != s[:-1]) / max(len(x) - 1, 1))
