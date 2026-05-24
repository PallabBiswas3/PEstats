from __future__ import annotations
import numpy as np
from scipy import stats as sp_stats
from dataclasses import dataclass

@dataclass
class RegressionResult:
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_error: float
    label: str = ""

    def summary(self) -> None:
        print(f"\nLinear Regression — '{self.label}'")
        print("─" * 40)
        print(f"  Slope      : {self.slope:.6f}")
        print(f"  Intercept  : {self.intercept:.6f}")
        print(f"  R²         : {self.r_squared:.6f}")
        print(f"  p-value    : {self.p_value:.6f}")
        print(f"  Std error  : {self.std_error:.6f}")
        print("─" * 40)

class RegressionMixin:

    def linear_regression(
        self, x: "np.ndarray | list | None" = None
    ) -> "RegressionMixin":
        """
        Fit a linear regression on self.data (as y).

        Parameters
        ----------
        x : independent variable array.
            If None, uses sample index (0, 1, 2, ...) — fits a trend line.
        """
        y = self.data.flatten()
        if x is None:
            x_arr = np.arange(len(y), dtype=float)
        else:
            x_arr = np.asarray(x, dtype=float).flatten()

        result = sp_stats.linregress(x_arr, y)

        self.regression_result = RegressionResult(
            slope      = float(result.slope),
            intercept  = float(result.intercept),
            r_squared  = float(result.rvalue ** 2),
            p_value    = float(result.pvalue),
            std_error  = float(result.stderr),
            label      = self.label,
        )

        self._record("linear_regression()")
        return self

    def detrend(self, method: str = "linear") -> "RegressionMixin":
        """
        Remove trend from self.data.
        Essential before spectral analysis of slowly drifting signals
        like baseline-drifted FTIR or slowly varying EEG.

        Parameters
        ----------
        method : 'linear'   — subtract best-fit line
                 'constant' — subtract mean only
        """
        from scipy.signal import detrend as sp_detrend
        self.data = sp_detrend(self.data, type=method)
        self._record(f"detrend(method={method})")
        return self