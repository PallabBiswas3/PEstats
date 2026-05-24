from __future__ import annotations
import numpy as np
from scipy import stats as sp_stats
from PBstats.statistics.base import DescriptiveResult

class DescriptiveMixin:

    def describe(self) -> "DescriptiveMixin":
        """
        Compute full descriptive statistics on self.data.
        Result stored in self.descriptive_result.
        self.data is unchanged — describe() is non-destructive.

        Includes RMS and SNR estimate — metrics standard stats
        libraries omit but which matter for signal data.
        """
        x = self.data.flatten().astype(float)
        x = x[~np.isnan(x)]     # exclude NaNs from stats

        mean    = float(np.mean(x))
        std     = float(np.std(x, ddof=1))
        rms     = float(np.sqrt(np.mean(x ** 2)))

        # SNR estimate: mean power / noise power
        # Noise estimated as std of the first-difference (removes signal trend)
        noise_est = float(np.std(np.diff(x), ddof=1)) / np.sqrt(2)
        snr_db = (
            20 * np.log10(rms / (noise_est + 1e-12))
            if noise_est > 0 else 0.0
        )

        q25, q75 = np.percentile(x, [25, 75])

        self.descriptive_result = DescriptiveResult(
            mean       = mean,
            median     = float(np.median(x)),
            std        = std,
            variance   = float(np.var(x, ddof=1)),
            skewness   = float(sp_stats.skew(x)),
            kurtosis   = float(sp_stats.kurtosis(x)),
            minimum    = float(x.min()),
            maximum    = float(x.max()),
            q25        = float(q25),
            q75        = float(q75),
            iqr        = float(q75 - q25),
            rms        = rms,
            snr        = float(snr_db),
            n          = len(x),
            label      = self.label,
        )

        self._record("describe()")
        return self

    def peak_to_peak(self) -> float:
        """Quick peak-to-peak amplitude. Non-destructive."""
        return float(self.data.max() - self.data.min())

    def energy(self) -> float:
        """Total signal energy (sum of squares). Non-destructive."""
        return float(np.sum(self.data ** 2))

    def zero_crossing_rate(self) -> float:
        """
        Fraction of samples where signal crosses zero.
        Useful for distinguishing voiced/unvoiced in audio,
        or detecting tremor in EMG signals.
        """
        x = self.data.flatten()
        crossings = np.sum(np.diff(np.sign(x)) != 0)
        return float(crossings / len(x))