from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import coherence as sp_coherence
from scipy.signal import correlate, correlation_lags

from PBstats.core.steps import step
from PBstats.core.utils import require_1d
from PBstats.statistics.base import (
    CoherenceResult,
    CorrelationResult,
    CrossCorrelationResult,
)


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation function via FFT — O(n log n) instead of O(n·lags)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    nfft = next_fast_len(2 * n - 1)
    f = rfft(x, nfft)
    r = irfft(f * np.conj(f), nfft)[: max_lag + 1]
    return r / (r[0] + 1e-300)


class CorrelationMixin:

    @step
    def correlate(
        self,
        signals: "list[np.ndarray]",
        labels: "list[str] | None" = None,
        method: str = "pearson",
    ) -> "CorrelationMixin":
        """
        Pairwise correlation between ``self.data`` and other signals.

        method : 'pearson' (linear) | 'spearman' (rank, robust) |
                 'kendall' (rank concordance, small n)
        Signals of different length are compared over their common prefix.
        """
        corr_fn = {
            "pearson": sp_stats.pearsonr,
            "spearman": sp_stats.spearmanr,
            "kendall": sp_stats.kendalltau,
        }.get(method)
        if corr_fn is None:
            raise ValueError(f"Unknown method '{method}'. Choose: pearson, spearman, kendall")

        all_signals = [self.data.flatten()] + [np.asarray(s, float).flatten() for s in signals]
        all_labels = [self.label or "self"] + (
            list(labels) if labels else [f"signal_{i}" for i in range(len(signals))]
        )
        n = len(all_signals)
        corr = np.eye(n)
        pairwise = []
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    r, p = 1.0, 0.0
                else:
                    m = min(len(all_signals[i]), len(all_signals[j]))
                    r, p = corr_fn(all_signals[i][:m], all_signals[j][:m])[:2]
                    corr[i, j] = corr[j, i] = r
                pairwise.append({"label_a": all_labels[i], "label_b": all_labels[j],
                                 "r": float(r), "p": float(p)})

        self.correlation_result = CorrelationResult(
            method=method, matrix=corr, labels=all_labels, pairwise=pairwise,
        )
        self._record(f"correlate(method={method}, n_signals={len(signals)})")
        return self

    @step
    def autocorrelation(self, max_lag: int | None = None) -> "CorrelationMixin":
        """
        Autocorrelation at lags 0..max_lag (default n // 4), stored in
        ``self.autocorr``. Reveals periodicity; also checks regression
        residuals for independence. FFT-based, fast on long signals.
        """
        x = require_1d(self.data, "autocorrelation")
        x = x[~np.isnan(x)]
        lag = int(min(max_lag or len(x) // 4, len(x) - 1))
        self.autocorr = acf(x, lag)
        self.autocorr_n = len(x)
        self._record(f"autocorrelation(max_lag={lag})")
        return self

    @step
    def cross_correlation(self, other, max_lag: int | None = None) -> "CorrelationMixin":
        """
        Normalised cross-correlation with another signal — estimates the time
        delay between two sensors / channels. Result in ``self.xcorr_result``;
        ``.best_lag`` > 0 means ``self`` lags (is a delayed copy of) ``other``.
        """
        x = require_1d(self.data, "cross_correlation").astype(float)
        y = np.asarray(other, dtype=float).flatten()
        x = (x - x.mean()) / (x.std() + 1e-300)
        y = (y - y.mean()) / (y.std() + 1e-300)
        values = correlate(x, y, mode="full", method="fft") / min(len(x), len(y))
        lags = correlation_lags(len(x), len(y), mode="full")
        if max_lag is not None:
            keep = np.abs(lags) <= max_lag
            lags, values = lags[keep], values[keep]

        self.xcorr_result = CrossCorrelationResult(
            lags=lags, values=values, fs=self.fs, label=self.label,
        )
        self._record(f"cross_correlation(best_lag={self.xcorr_result.best_lag})")
        return self

    @step
    def coherence(self, other, nperseg: int | None = None) -> "CorrelationMixin":
        """
        Magnitude-squared coherence with another signal: a frequency-resolved
        correlation (0 = unrelated, 1 = perfectly linearly related at that
        frequency). Standard for EEG connectivity and vibration transfer paths.
        Result in ``self.coherence_result``.
        """
        self._require_fs("coherence")
        x = require_1d(self.data, "coherence")
        y = np.asarray(other, dtype=float).flatten()
        m = min(len(x), len(y))
        nperseg = int(min(nperseg or max(int(2 * self.fs), 8), m))
        freqs, cxy = sp_coherence(x[:m], y[:m], fs=self.fs, nperseg=nperseg)
        self.coherence_result = CoherenceResult(
            freqs=freqs, coherence=cxy, fs=self.fs, label=self.label,
        )
        self._record(f"coherence(nperseg={nperseg})")
        return self
