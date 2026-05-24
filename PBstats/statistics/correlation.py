from __future__ import annotations
import numpy as np
from scipy import stats as sp_stats
from PBstats.statistics.base import CorrelationResult

class CorrelationMixin:

    def correlate(
        self,
        signals: "list[np.ndarray]",
        labels: "list[str] | None" = None,
        method: str = "pearson",
    ) -> "CorrelationMixin":
        """
        Compute pairwise correlation between self.data and a list of signals.

        Parameters
        ----------
        signals : list of arrays to correlate against self.data
        labels  : names for each signal (used in output)
        method  : 'pearson'  — linear correlation (assumes normality)
                  'spearman' — rank-based (robust, non-parametric)
                  'kendall'  — rank concordance (better for small n)
        """
        all_signals = [self.data.flatten()] + [
            np.asarray(s).flatten() for s in signals
        ]
        all_labels = [self.label] + (
            labels if labels else [f"signal_{i}" for i in range(len(signals))]
        )

        n = len(all_signals)
        corr_matrix = np.zeros((n, n))
        pairwise    = []

        corr_fn = {
            "pearson":  sp_stats.pearsonr,
            "spearman": sp_stats.spearmanr,
            "kendall":  sp_stats.kendalltau,
        }.get(method)

        if corr_fn is None:
            raise ValueError(
                f"Unknown method '{method}'. Choose: pearson, spearman, kendall"
            )

        for i in range(n):
            for j in range(n):
                min_len = min(len(all_signals[i]), len(all_signals[j]))
                r, p = corr_fn(all_signals[i][:min_len], all_signals[j][:min_len])
                corr_matrix[i, j] = r
                if j >= i:
                    pairwise.append({
                        "label_a": all_labels[i],
                        "label_b": all_labels[j],
                        "r": float(r),
                        "p": float(p),
                    })

        self.correlation_result = CorrelationResult(
            method=method,
            matrix=corr_matrix,
            labels=all_labels,
            pairwise=pairwise,
        )

        self._record(f"correlate(method={method}, n_signals={len(signals)})")
        return self

    def autocorrelation(self, max_lag: int = None) -> "CorrelationMixin":
        """
        Compute autocorrelation of self.data.
        Useful for detecting periodicity and checking
        residuals in regression for independence.

        Result stored in self.autocorr — a 1D array of
        correlation coefficients at lags 0..max_lag.
        """
        x   = self.data.flatten()
        x   = (x - x.mean()) / (x.std() + 1e-12)
        n   = len(x)
        lag = max_lag or n // 4

        self.autocorr = np.array([
            float(np.corrcoef(x[:n - k], x[k:])[0, 1])
            for k in range(lag + 1)
        ])

        self._record(f"autocorrelation(max_lag={lag})")
        return self