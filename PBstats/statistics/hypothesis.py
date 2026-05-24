from __future__ import annotations
import numpy as np
from scipy import stats as sp_stats
from PBstats.statistics.base import HypothesisResult

class HypothesisMixin:

    def test_normality(
        self, method: str = "shapiro", alpha: float = 0.05
    ) -> "HypothesisMixin":
        """
        Test whether self.data is normally distributed.

        Parameters
        ----------
        method : 'shapiro'  — Shapiro-Wilk (best for n < 5000)
                 'kstest'   — Kolmogorov-Smirnov against normal
                 'dagostino'— D'Agostino K² (good for n > 20)
        alpha  : significance level (default 0.05)
        """
        x = self.data.flatten()

        if method == "shapiro":
            stat, p = sp_stats.shapiro(x[:5000])   # Shapiro limited to 5000
            note = "H0: data is normally distributed"

        elif method == "kstest":
            x_std = (x - x.mean()) / (x.std() + 1e-12)
            stat, p = sp_stats.kstest(x_std, "norm")
            note = "H0: data follows standard normal distribution"

        elif method == "dagostino":
            stat, p = sp_stats.normaltest(x)
            note = "H0: data is normally distributed (skew + kurtosis test)"

        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose: shapiro, kstest, dagostino"
            )

        self.normality_result = HypothesisResult(
            test=f"Normality ({method})",
            statistic=float(stat),
            p_value=float(p),
            alpha=alpha,
            reject_null=bool(p < alpha),
            interpretation=note,
            label=self.label,
        )

        self._record(f"test_normality(method={method})")
        return self

    def test_stationarity(
        self, method: str = "adf", alpha: float = 0.05
    ) -> "HypothesisMixin":
        """
        Test whether a time series is stationary.
        Critical before applying FFT or spectral analysis —
        non-stationary signals produce misleading spectra.

        Parameters
        ----------
        method : 'adf'  — Augmented Dickey-Fuller (tests for unit root)
                 'kpss' — KPSS (tests for trend stationarity)
        """
        try:
            from statsmodels.tsa.stattools import adfuller, kpss
        except ImportError:
            raise ImportError(
                "statsmodels required for stationarity tests: "
                "pip install statsmodels"
            )

        x = self.data.flatten()

        if method == "adf":
            out   = adfuller(x, autolag="AIC")
            stat, p = out[0], out[1]
            note  = "H0: series has a unit root (non-stationary)"

        elif method == "kpss":
            out   = kpss(x, regression="c", nlags="auto")
            stat, p = out[0], out[1]
            note  = "H0: series is stationary around a constant"

        else:
            raise ValueError(f"Unknown method '{method}'. Choose: adf, kpss")

        self.stationarity_result = HypothesisResult(
            test=f"Stationarity ({method})",
            statistic=float(stat),
            p_value=float(p),
            alpha=alpha,
            reject_null=bool(p < alpha),
            interpretation=note,
            label=self.label,
        )

        self._record(f"test_stationarity(method={method})")
        return self

    def ttest(
        self,
        other: "np.ndarray | list",
        paired: bool = False,
        alpha: float = 0.05,
    ) -> "HypothesisMixin":
        """
        Compare self.data mean against another signal or a scalar.

        Parameters
        ----------
        other  : array to compare against, or a scalar (one-sample t-test)
        paired : if True, use paired t-test (same subjects, two conditions)
        alpha  : significance level
        """
        x = self.data.flatten()

        if np.isscalar(other):
            stat, p = sp_stats.ttest_1samp(x, popmean=float(other))
            test_name = f"One-sample t-test (vs {other})"
        elif paired:
            stat, p = sp_stats.ttest_rel(x, np.asarray(other).flatten())
            test_name = "Paired t-test"
        else:
            stat, p = sp_stats.ttest_ind(x, np.asarray(other).flatten())
            test_name = "Independent t-test"

        self.ttest_result = HypothesisResult(
            test=test_name,
            statistic=float(stat),
            p_value=float(p),
            alpha=alpha,
            reject_null=bool(p < alpha),
            interpretation="H0: means are equal",
            label=self.label,
        )

        self._record(f"ttest(paired={paired})")
        return self

    def mannwhitney(
        self, other: "np.ndarray | list", alpha: float = 0.05
    ) -> "HypothesisMixin":
        """
        Non-parametric alternative to t-test.
        Use when normality test rejects H0 — does not assume normal distribution.
        """
        stat, p = sp_stats.mannwhitneyu(
            self.data.flatten(),
            np.asarray(other).flatten(),
            alternative="two-sided",
        )

        self.mannwhitney_result = HypothesisResult(
            test="Mann-Whitney U",
            statistic=float(stat),
            p_value=float(p),
            alpha=alpha,
            reject_null=bool(p < alpha),
            interpretation="H0: distributions are equal (non-parametric)",
            label=self.label,
        )

        self._record("mannwhitney()")
        return self