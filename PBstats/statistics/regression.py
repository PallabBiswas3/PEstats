from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

from PBstats.core.steps import step
from PBstats.core.utils import per_channel, safe_print


@dataclass
class RegressionResult:
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_error: float
    label: str = ""

    def predict(self, x) -> np.ndarray:
        return self.intercept + self.slope * np.asarray(x, dtype=float)

    def summary(self) -> None:
        safe_print(f"\nLinear Regression — '{self.label}'")
        safe_print("─" * 40)
        safe_print(f"  Slope      : {self.slope:.6f}")
        safe_print(f"  Intercept  : {self.intercept:.6f}")
        safe_print(f"  R²         : {self.r_squared:.6f}")
        safe_print(f"  p-value    : {self.p_value:.6g}")
        safe_print(f"  Std error  : {self.std_error:.6f}")
        safe_print("─" * 40)


class RegressionMixin:

    @step
    def linear_regression(self, x: "np.ndarray | list | None" = None) -> "RegressionMixin":
        """
        Fit y = slope·x + intercept with ``self.data`` as y.
        x defaults to time in seconds (or sample index without fs) — i.e. a
        trend line, whose slope is then in units per second.
        """
        y = self.data.flatten().astype(float)
        if x is None:
            x_arr = np.arange(len(y), dtype=float) / (self.fs or 1.0)
        else:
            x_arr = np.asarray(x, dtype=float).flatten()
        ok = ~(np.isnan(y) | np.isnan(x_arr))
        res = sp_stats.linregress(x_arr[ok], y[ok])

        self.regression_result = RegressionResult(
            slope=float(res.slope), intercept=float(res.intercept),
            r_squared=float(res.rvalue ** 2), p_value=float(res.pvalue),
            std_error=float(res.stderr), label=self.label,
        )
        self._record("linear_regression()")
        return self

    @step
    def detrend(self, method: str = "linear", order: int = 2) -> "RegressionMixin":
        """
        Remove a trend from every channel — essential before spectral
        analysis of drifting signals (FTIR baselines, slow EEG drift).

        method : 'linear'   — subtract best-fit line
                 'constant' — subtract the mean
                 'poly'     — subtract a polynomial of degree ``order``
                              (curved baselines)
        """
        if method in ("linear", "constant"):
            from scipy.signal import detrend as sp_detrend
            self.data = sp_detrend(self.data, type=method, axis=0)
        elif method == "poly":
            def _poly(c):
                t = np.linspace(-1, 1, len(c))  # well-conditioned
                return c - np.polyval(np.polyfit(t, c, order), t)
            self.data = per_channel(_poly, self.data)
        else:
            raise ValueError("method must be 'linear', 'constant' or 'poly'")
        self._record(f"detrend(method={method})")
        return self
