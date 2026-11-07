from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from PBstats.core.steps import step
from PBstats.statistics.base import HypothesisResult


def _clean(a) -> np.ndarray:
    a = np.asarray(a, dtype=float).flatten()
    return a[~np.isnan(a)]


def _cohens_d(a: np.ndarray, b: np.ndarray, paired: bool) -> float:
    if paired:
        d = a - b
        return float(d.mean() / (d.std(ddof=1) + 1e-300))
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / (pooled + 1e-300))


class HypothesisMixin:

    @step
    def test_normality(self, method: str = "shapiro", alpha: float = 0.05) -> "HypothesisMixin":
        """
        Is ``self.data`` normally distributed?

        method : 'shapiro'   — Shapiro-Wilk. For n > 5000 a reproducible
                               *random* subsample of 5000 is tested (the first
                               5000 samples of a time series are not
                               representative of the whole).
                 'kstest'    — Kolmogorov-Smirnov vs. a fitted normal
                 'dagostino' — D'Agostino K² (skew + kurtosis), n > 20
        """
        x = _clean(self.data)

        if method == "shapiro":
            if len(x) > 5000:
                x = np.random.default_rng(0).choice(x, 5000, replace=False)
            stat, p = sp_stats.shapiro(x)
            note = "H0: data is normally distributed"
        elif method == "kstest":
            x_std = (x - x.mean()) / (x.std(ddof=1) + 1e-12)
            stat, p = sp_stats.kstest(x_std, "norm")
            note = "H0: data follows a normal distribution"
        elif method == "dagostino":
            stat, p = sp_stats.normaltest(x)
            note = "H0: data is normally distributed (skew + kurtosis test)"
        else:
            raise ValueError(f"Unknown method '{method}'. Choose: shapiro, kstest, dagostino")

        self.normality_result = HypothesisResult(
            test=f"Normality ({method})", statistic=float(stat), p_value=float(p),
            alpha=alpha, reject_null=bool(p < alpha), interpretation=note,
            label=self.label,
        )
        self._record(f"test_normality(method={method})")
        return self

    @step
    def test_stationarity(self, method: str = "adf", alpha: float = 0.05) -> "HypothesisMixin":
        """
        Is the time series stationary? Check before trusting a single FFT —
        non-stationary signals produce misleading spectra (use stft/wavelet).

        method : 'adf'  — Augmented Dickey-Fuller (H0: unit root / non-stationary)
                 'kpss' — KPSS (H0: stationary around a constant)
        Note the opposite null hypotheses: for ADF, rejecting H0 means
        stationary; for KPSS, rejecting H0 means non-stationary.
        """
        import warnings

        from statsmodels.tsa.stattools import adfuller, kpss

        x = _clean(self.data)
        if method == "adf":
            stat, p = adfuller(x, autolag="AIC")[:2]
            note = "H0: series has a unit root (non-stationary)"
        elif method == "kpss":
            with warnings.catch_warnings():
                # p-value is clipped to the table range [0.01, 0.1]; that's expected
                warnings.simplefilter("ignore")
                stat, p = kpss(x, regression="c", nlags="auto")[:2]
            note = "H0: series is stationary around a constant"
        else:
            raise ValueError(f"Unknown method '{method}'. Choose: adf, kpss")

        self.stationarity_result = HypothesisResult(
            test=f"Stationarity ({method})", statistic=float(stat), p_value=float(p),
            alpha=alpha, reject_null=bool(p < alpha), interpretation=note,
            label=self.label,
        )
        self._record(f"test_stationarity(method={method})")
        return self

    @step
    def ttest(
        self, other: "np.ndarray | list | float", paired: bool = False, alpha: float = 0.05,
    ) -> "HypothesisMixin":
        """
        Compare the mean of ``self.data`` with another sample or a scalar.
        Reports Cohen's d as effect size (statistical significance alone says
        nothing about how *big* a difference is — with long signals almost
        everything is "significant").

        paired : paired t-test (same subjects, two conditions)
        Independent samples use Welch's t-test (no equal-variance assumption).
        """
        x = _clean(self.data)
        if np.isscalar(other):
            stat, p = sp_stats.ttest_1samp(x, popmean=float(other))
            name = f"One-sample t-test (vs {other})"
            d = float((x.mean() - float(other)) / (x.std(ddof=1) + 1e-300))
        else:
            y = _clean(other)
            if paired:
                stat, p = sp_stats.ttest_rel(x, y)
                name = "Paired t-test"
            else:
                stat, p = sp_stats.ttest_ind(x, y, equal_var=False)
                name = "Welch t-test"
            d = _cohens_d(x, y, paired)

        self.ttest_result = HypothesisResult(
            test=name, statistic=float(stat), p_value=float(p), alpha=alpha,
            reject_null=bool(p < alpha), interpretation="H0: means are equal",
            label=self.label, effect_size=d, effect_name="Cohen's d",
        )
        self._record(f"ttest(paired={paired})")
        return self

    @step
    def mannwhitney(self, other: "np.ndarray | list", alpha: float = 0.05) -> "HypothesisMixin":
        """
        Non-parametric alternative to the t-test (no normality assumption).
        Reports the rank-biserial correlation as effect size.
        """
        x, y = _clean(self.data), _clean(other)
        stat, p = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
        rbc = 2.0 * float(stat) / (len(x) * len(y)) - 1.0

        self.mannwhitney_result = HypothesisResult(
            test="Mann-Whitney U", statistic=float(stat), p_value=float(p), alpha=alpha,
            reject_null=bool(p < alpha),
            interpretation="H0: distributions are equal (non-parametric)",
            label=self.label, effect_size=rbc, effect_name="rank-biserial r",
        )
        self._record("mannwhitney()")
        return self
