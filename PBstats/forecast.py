"""
Forecasting with (S)ARIMA models from statsmodels.

    d.forecast(horizon=5.0)                  # 5 s ahead, order chosen by AIC
    d.forecast(steps=100, holdout=100)       # honest check: fit without the
                                             # last 100 samples, score on them
    d.forecast_result.plot()
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from PBstats.core.steps import step
from PBstats.core.utils import require_1d, safe_print


@dataclass
class ForecastResult:
    times: np.ndarray              # time of each forecast point (s, or samples without fs)
    mean: np.ndarray
    lower: np.ndarray              # prediction interval
    upper: np.ndarray
    order: tuple
    seasonal_order: tuple
    aic: float
    level: float                   # interval coverage, e.g. 0.95
    history_times: np.ndarray = field(repr=False)
    history: np.ndarray = field(repr=False)
    actual: Optional[np.ndarray] = field(default=None, repr=False)   # held-out truth
    trend: str = "c"
    label: str = ""

    @property
    def metrics(self) -> dict:
        """MAE / RMSE / MAPE and interval coverage on the held-out samples."""
        if self.actual is None:
            return {}
        err = self.actual - self.mean
        nz = np.abs(self.actual) > 1e-12
        mae = float(np.mean(np.abs(err)))
        naive = float(np.mean(np.abs(self.actual - self.history[-1])))   # last-value forecast
        return {
            "MAE": mae,
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAPE_%": float(100 * np.mean(np.abs(err[nz] / self.actual[nz]))) if nz.any() else float("nan"),
            "coverage": float(np.mean((self.actual >= self.lower) & (self.actual <= self.upper))),
            "MAE_naive": naive,
            # > 0: better than repeating the last value; <= 0: the model adds nothing
            "skill": 1.0 - mae / naive if naive > 0 else float("nan"),
        }

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({"time": self.times, "forecast": self.mean,
                           "lower": self.lower, "upper": self.upper})
        if self.actual is not None:
            df["actual"] = self.actual
        return df

    def summary(self) -> None:
        safe_print(f"\nForecast — '{self.label}'  ARIMA{self.order}"
                   + (f"x{self.seasonal_order}" if any(self.seasonal_order) else "")
                   + f"  AIC={self.aic:.1f}  horizon={len(self.mean)} samples")
        for k, v in self.metrics.items():
            safe_print(f"  {k:<9s}: {v:.4g}")

    def plot(self, save_path: Optional[str] = None):
        from PBstats.visualization.plots import plot_forecast
        return plot_forecast(self, save_path=save_path)


def _fit(y, order, seasonal_order, trend):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(y, order=order, seasonal_order=seasonal_order, trend=trend,
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


def _score_candidate(y, order, trend, seasonal, select, v, folds):
    """Validation MAE averaged over rolling origins (or AIC); inf on failure."""
    try:
        if select == "aic":
            return float(_fit(y, order, seasonal, trend).aic)
        errs = []
        for k in range(folds, 0, -1):
            cut = len(y) - k * v
            pred = np.asarray(_fit(y[:cut], order, seasonal, trend).forecast(v))
            errs.append(np.mean(np.abs(pred - y[cut:cut + v])))
        score = float(np.mean(errs))
        return score if np.isfinite(score) else float("inf")
    except Exception:
        return float("inf")


def _candidates():
    """(p, d, q) up to (3, 1, 2), each with the trend terms that make sense:
    constant or linear trend without differencing; none or drift with it."""
    for p, d, q in itertools.product(range(4), range(2), range(3)):
        if p == q == 0 and d == 0:
            continue
        for trend in (("c", "ct") if d == 0 else ("n", "t")):
            yield (p, d, q), trend


class ForecastMixin:

    @step
    def forecast(
        self,
        steps: Optional[int] = None,
        horizon: Optional[float] = None,
        order: Optional[tuple] = None,
        trend: Optional[str] = None,
        select: str = "validation",
        seasonal_period: Optional[int] = None,
        holdout: int = 0,
        max_history: int = 2000,
        level: float = 0.95,
        n_jobs: Optional[int] = None,
    ) -> "ForecastMixin":
        """
        Forecast future values with an ARIMA model. Result in
        ``self.forecast_result``; ``self.data`` is unchanged.

        Parameters
        ----------
        steps / horizon : how far ahead — samples, or seconds (needs fs)
        order           : (p, d, q). None = search a grid (p ≤ 3, d ≤ 1,
                          q ≤ 2, with constant / linear-trend / drift terms)
        trend           : statsmodels trend term when ``order`` is given
                          ('n', 'c', 't', 'ct'); default 'c' (d=0) or 'n'
        select          : how the grid search picks a model —
                          'validation' (default): rolling-origin validation —
                          forecast the last two stretches of the training data
                          from just before each, keep the model with the lowest
                          mean error, refit on everything. Targets forecast
                          accuracy; a refit that diverges falls back to the
                          next-best model.
                          'aic': lowest in-sample AIC (faster, but in testing
                          it often picked models that forecast badly).
        seasonal_period : add a seasonal (1, 0, 1) component with this
                          period in samples (e.g. 24 for hourly data with a
                          daily cycle)
        holdout         : fit without the last ``holdout`` samples and forecast
                          them, so ``forecast_result.metrics`` gives an honest
                          out-of-sample error (steps defaults to holdout)
        max_history     : fit on at most this many most-recent samples
                          (ARIMA cost grows with length; downsample long,
                          oversampled signals first)
        level           : prediction-interval coverage
        n_jobs          : parallel workers for the grid search (-1 = all
                          cores). A full search fits ~150 models: seconds to
                          tens of seconds; pass ``order`` to skip it.

        Notes
        -----
        Classical ARIMA suits smooth, autocorrelated series (sensor drift,
        physiological trends, demand). It will not forecast beat-by-beat ECG
        morphology — forecast derived series such as heart rate instead.
        """
        y_all = require_1d(self.data, "forecast").astype(float)
        if np.isnan(y_all).any():
            raise ValueError("Signal contains NaN — call remove_missing() first")
        if horizon is not None:
            self._require_fs("forecast(horizon=...)")
            steps = int(round(horizon * self.fs))
        steps = steps or holdout
        if not steps or steps < 1:
            raise ValueError("Give steps, horizon or holdout")
        if holdout and holdout >= len(y_all) - 20:
            raise ValueError("holdout leaves too little data to fit")

        train = y_all[: len(y_all) - holdout] if holdout else y_all
        start = max(0, len(train) - max_history)
        y = train[start:]
        seasonal = (1, 0, 1, seasonal_period) if seasonal_period else (0, 0, 0, 0)

        if order is None:
            if select not in ("validation", "aic"):
                raise ValueError("select must be 'validation' or 'aic'")
            from joblib import Parallel, delayed

            folds = 2
            v = max(1, min(steps, len(y) // (3 * folds)))
            cands = list(_candidates())
            scores = Parallel(n_jobs=n_jobs)(
                delayed(_score_candidate)(y, o, tr, seasonal, select, v, folds) for o, tr in cands
            )
            ranked = [c for _, c in sorted(zip(scores, cands), key=lambda z: z[0])
                      if np.isfinite(_)]
            if not ranked:
                raise RuntimeError("No ARIMA model could be fitted")
            # Refit in rank order; skip any whose full-data forecast diverges.
            span = float(np.ptp(y)) or 1.0
            res = None
            for order, trend in ranked[:10]:
                try:
                    cand = _fit(y, order, seasonal, trend)
                    pred = np.asarray(cand.forecast(steps))
                except Exception:
                    continue
                if np.all(np.isfinite(pred)) and np.max(np.abs(pred - y[-1])) < 10 * span:
                    res = cand
                    break
            if res is None:
                raise RuntimeError("Every candidate model diverged — try order=... or more data")
        else:
            order = tuple(order)
            trend = trend or ("c" if order[1] == 0 else "n")
            res = _fit(y, order, seasonal, trend)

        fc = res.get_forecast(steps)
        ci = np.asarray(fc.conf_int(alpha=1 - level))
        scale = self.fs or 1.0
        t_end = len(train)
        self.forecast_result = ForecastResult(
            times=np.arange(t_end, t_end + steps) / scale,
            mean=np.asarray(fc.predicted_mean),
            lower=ci[:, 0], upper=ci[:, 1],
            order=tuple(order), seasonal_order=seasonal, aic=float(res.aic), level=level,
            history_times=np.arange(start, t_end) / scale, history=y,
            actual=y_all[t_end:t_end + steps] if holdout else None,
            trend=trend, label=self.label,
        )
        if holdout and steps > holdout:
            self.forecast_result.actual = None   # can't score beyond the held-out data
        self._record(f"forecast(steps={steps}, order={tuple(order)}, trend={trend})")
        return self
