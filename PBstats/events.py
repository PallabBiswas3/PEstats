"""
Event & anomaly detection: where does the signal's behaviour change, and
which stretches look unlike the rest?

    d.detect_change_points(cost="meanvar")      # PELT segmentation
    d.change_point_result.segments              # start/end/mean/std per regime
    d.anomaly_scores(window=5)                  # IsolationForest on windowed features
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from PBstats.core.steps import step
from PBstats.core.utils import require_1d


@dataclass
class ChangePointResult:
    indices: np.ndarray          # change-point sample indices (segment starts, excl. 0)
    fs: float | None
    cost: str
    penalty: float
    segments: pd.DataFrame = field(repr=False)   # start_s, end_s, n, mean, std
    label: str = ""

    @property
    def times(self) -> np.ndarray:
        """Change points in seconds (samples without fs)."""
        return self.indices / (self.fs or 1.0)

    def __len__(self) -> int:
        return len(self.indices)


def pelt(
    x: np.ndarray,
    cost: str = "meanvar",
    penalty: float | None = None,
    min_size: int = 2,
    max_points: int = 10000,
) -> tuple[np.ndarray, float]:
    """
    PELT exact segmentation (Killick, Fearnhead & Eckley, JASA 2012) with a
    Gaussian likelihood cost.

    cost     : 'mean' — changes in level (variance assumed constant)
               'meanvar' — changes in level and/or variability
    penalty  : per-change-point penalty; default BIC-style
               (2 params for 'mean', 3 for 'meanvar') × log(n)
    min_size : minimum segment length in samples
    max_points : signals longer than this are segmented on a block grid
               (change points can only fall on block boundaries), using exact
               segment statistics from cumulative sums — so long recordings
               stay fast without any approximation of the cost.

    Returns (change-point indices, penalty used).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if cost not in ("mean", "meanvar"):
        raise ValueError("cost must be 'mean' or 'meanvar'")
    if np.isnan(x).any():
        raise ValueError("Signal contains NaN — call remove_missing() first")

    block = int(np.ceil(n / max_points)) if n > max_points else 1
    grid = np.unique(np.r_[np.arange(0, n, block), n])
    s1 = np.r_[0.0, np.cumsum(x)][grid]
    s2 = np.r_[0.0, np.cumsum(x * x)][grid]
    m = len(grid) - 1
    min_blocks = max(1, int(np.ceil(min_size / block)))

    # Robust noise scale from first differences (for the 'mean' cost).
    d = np.diff(x)
    sigma2 = (1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) ** 2 if len(d) else 0.0
    sigma2 = sigma2 if sigma2 > 0 else (np.var(x) or 1.0)
    var_floor = 1e-6 * (np.var(x) or 1.0)

    if penalty is None:
        penalty = (2 if cost == "mean" else 3) * np.log(n)

    def seg_cost(i: np.ndarray, j: int) -> np.ndarray:
        cnt = grid[j] - grid[i]
        a = s1[j] - s1[i]
        b = s2[j] - s2[i]
        if cost == "mean":
            return (b - a * a / cnt) / sigma2
        var = np.maximum(b / cnt - (a / cnt) ** 2, var_floor)
        return cnt * (np.log(2 * np.pi * var) + 1)

    # F[t]: optimal penalised cost of x[:grid[t]]; R: candidate last change points.
    F = np.full(m + 1, np.inf)
    F[0] = -penalty
    last = np.zeros(m + 1, dtype=int)
    R = np.array([0])
    for t in range(1, m + 1):
        eligible = t - R >= min_blocks
        cand = R[eligible]
        if len(cand):
            vals = F[cand] + seg_cost(cand, t)
            k = int(np.argmin(vals))
            F[t] = vals[k] + penalty
            last[t] = cand[k]
            # PELT pruning: a candidate that is already worse than the optimum
            # (without the penalty) can never become optimal later.
            R = np.r_[R[~eligible], cand[vals <= F[t]]]
        if np.isfinite(F[t]):
            R = np.r_[R, t]

    cps, t = [], m
    while t > 0:
        t = last[t]
        if t > 0:
            cps.append(grid[t])
    return np.array(sorted(cps), dtype=int), float(penalty)


class EventsMixin:

    @step
    def detect_change_points(
        self,
        cost: str = "meanvar",
        penalty: float | None = None,
        min_duration: float | None = None,
        max_points: int = 10000,
    ) -> "EventsMixin":
        """
        Find the points where the signal's statistical behaviour changes
        (PELT segmentation). Result in ``self.change_point_result``.

        Typical uses: regime changes in sensor data, onset of activity in
        EMG/accelerometry, sleep-stage-like transitions, machine state changes.
        For changes in *rhythm* rather than level, run it on a derived
        signal (e.g. the Hilbert envelope or a windowed feature).

        Parameters
        ----------
        cost         : 'meanvar' (default; level and/or variability) or 'mean'
        penalty      : larger → fewer change points (default: BIC)
        min_duration : minimum segment length in seconds (default: 1% of
                       the record, at least 2 samples)
        """
        x = require_1d(self.data, "detect_change_points")
        scale = self.fs or 1.0
        min_size = int(round(min_duration * scale)) if min_duration else max(2, len(x) // 100)
        idx, pen = pelt(x, cost=cost, penalty=penalty, min_size=min_size, max_points=max_points)
        bounds = np.r_[0, idx, len(x)]
        segs = pd.DataFrame([
            {"start_s": a / scale, "end_s": b / scale, "n": b - a,
             "mean": float(x[a:b].mean()), "std": float(x[a:b].std())}
            for a, b in zip(bounds[:-1], bounds[1:])
        ])
        self.change_point_result = ChangePointResult(
            indices=idx, fs=self.fs, cost=cost, penalty=pen, segments=segs, label=self.label,
        )
        self._record(f"detect_change_points(n={len(idx)}, cost={cost})")
        return self

    def anomaly_scores(
        self,
        window: float,
        step: float | None = None,
        include: list[str] | None = None,
        contamination: float | str = "auto",
        random_state: int = 0,
    ) -> pd.DataFrame:
        """
        Unsupervised anomaly detection: slice the signal into windows,
        extract features from each, and score how unusual each window is
        with an Isolation Forest. Finds artefacts, seizures, faults or rare
        events without labels.

        Parameters
        ----------
        window, step  : window length / hop in seconds (default hop = window)
        include       : feature categories (default: time + freq)
        contamination : expected share of anomalies, or 'auto'

        Returns
        -------
        DataFrame: t_start, t_end, score (higher = more anomalous), anomaly (bool).
        Also stored in ``self.anomaly_result``.
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler

        include = include or (["time", "freq"] if self.fs else ["time"])
        feats = self.windowed_features(window=window, step=step, include=include)
        X = feats.drop(columns=["label", "t_start", "t_end"]).to_numpy(dtype=float)
        if len(X) < 8:
            raise ValueError(f"Only {len(X)} windows — use a shorter window for anomaly scoring")
        X = SimpleImputer(strategy="median").fit_transform(X)
        X = RobustScaler().fit_transform(X)
        forest = IsolationForest(contamination=contamination, random_state=random_state).fit(X)
        out = feats[["t_start", "t_end"]].copy()
        out["score"] = -forest.score_samples(X)
        out["anomaly"] = forest.predict(X) == -1
        self.anomaly_result = out
        self._record(f"anomaly_scores(window={window}, n_anomalies={int(out.anomaly.sum())})")
        return out
