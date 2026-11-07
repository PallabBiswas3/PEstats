from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from PBstats.core.utils import safe_print


@dataclass
class DescriptiveResult:
    mean: float
    median: float
    std: float
    variance: float
    skewness: float
    kurtosis: float
    minimum: float
    maximum: float
    q25: float
    q75: float
    iqr: float
    rms: float          # root mean square — signal energy
    snr: float          # heuristic SNR estimate (dB)
    n: int              # sample count (NaNs excluded)
    label: str = ""
    mad: float = float("nan")                              # median absolute deviation
    values: np.ndarray = field(default=None, repr=False)   # the data described

    def summary(self) -> None:
        safe_print(f"\nDescriptive Stats — '{self.label}'")
        safe_print("─" * 40)
        safe_print(f"  N          : {self.n}")
        safe_print(f"  Mean       : {self.mean:.6f}")
        safe_print(f"  Median     : {self.median:.6f}")
        safe_print(f"  Std        : {self.std:.6f}")
        safe_print(f"  MAD        : {self.mad:.6f}")
        safe_print(f"  Skewness   : {self.skewness:.6f}")
        safe_print(f"  Kurtosis   : {self.kurtosis:.6f}")
        safe_print(f"  Min / Max  : {self.minimum:.4f} / {self.maximum:.4f}")
        safe_print(f"  Q25 / Q75  : {self.q25:.4f} / {self.q75:.4f}")
        safe_print(f"  IQR        : {self.iqr:.6f}")
        safe_print(f"  RMS        : {self.rms:.6f}")
        safe_print(f"  SNR (est.) : {self.snr:.2f} dB")
        safe_print("─" * 40)

    def to_dict(self) -> dict:
        return {
            "label": self.label, "n": self.n,
            "mean": self.mean, "median": self.median,
            "std": self.std, "variance": self.variance, "mad": self.mad,
            "skewness": self.skewness, "kurtosis": self.kurtosis,
            "min": self.minimum, "max": self.maximum,
            "q25": self.q25, "q75": self.q75,
            "iqr": self.iqr, "rms": self.rms, "snr_db": self.snr,
        }


@dataclass
class HypothesisResult:
    test: str
    statistic: float
    p_value: float
    alpha: float
    reject_null: bool
    interpretation: str
    label: str = ""
    effect_size: Optional[float] = None
    effect_name: str = ""

    def summary(self) -> None:
        verdict = "REJECT H0" if self.reject_null else "FAIL TO REJECT H0"
        safe_print(f"\nHypothesis Test — {self.test}")
        safe_print("─" * 40)
        safe_print(f"  Statistic  : {self.statistic:.6f}")
        safe_print(f"  p-value    : {self.p_value:.6g}")
        safe_print(f"  Alpha      : {self.alpha}")
        safe_print(f"  Decision   : {verdict}")
        if self.effect_size is not None:
            safe_print(f"  Effect     : {self.effect_name} = {self.effect_size:+.4f} "
                  f"({_effect_magnitude(self.effect_name, self.effect_size)})")
        safe_print(f"  Note       : {self.interpretation}")
        safe_print("─" * 40)


def _effect_magnitude(name: str, value: float) -> str:
    """Conventional verbal labels (Cohen 1988)."""
    v = abs(value)
    cuts = (0.2, 0.5, 0.8) if name.startswith("Cohen") else (0.1, 0.3, 0.5)
    if v < cuts[0]:
        return "negligible"
    if v < cuts[1]:
        return "small"
    if v < cuts[2]:
        return "medium"
    return "large"


@dataclass
class CorrelationResult:
    method: str
    matrix: np.ndarray            # (n_signals × n_signals)
    labels: list[str]
    pairwise: list[dict]          # flat list of {label_a, label_b, r, p}

    def strongest(self, n: int = 3) -> list[dict]:
        """The n strongest correlations by |r|, excluding self-pairs."""
        ranked = sorted(
            [p for p in self.pairwise if p["label_a"] != p["label_b"]],
            key=lambda x: abs(x["r"]),
            reverse=True,
        )
        return ranked[:n]

    def summary(self) -> None:
        safe_print(f"\nCorrelation ({self.method})")
        safe_print("─" * 40)
        for pair in self.strongest(5):
            safe_print(f"  {pair['label_a']:15s} × {pair['label_b']:15s} "
                  f"r={pair['r']:+.4f}  p={pair['p']:.4f}")
        safe_print("─" * 40)


@dataclass
class CrossCorrelationResult:
    lags: np.ndarray        # in samples
    values: np.ndarray      # normalised cross-correlation in [-1, 1]
    fs: Optional[float]
    label: str = ""

    @property
    def best_lag(self) -> int:
        """Lag (samples) of maximum |correlation|. Positive = self lags other."""
        return int(self.lags[np.argmax(np.abs(self.values))])

    @property
    def best_lag_seconds(self) -> float:
        if not self.fs:
            raise ValueError("fs is not set; use best_lag (samples)")
        return self.best_lag / self.fs

    @property
    def peak(self) -> float:
        return float(self.values[np.argmax(np.abs(self.values))])


@dataclass
class CoherenceResult:
    freqs: np.ndarray
    coherence: np.ndarray   # magnitude-squared coherence in [0, 1]
    fs: float
    label: str = ""

    def band_mean(self, low: float, high: float) -> float:
        mask = (self.freqs >= low) & (self.freqs <= high)
        return float(self.coherence[mask].mean())

    def peak_frequency(self) -> float:
        return float(self.freqs[np.argmax(self.coherence)])
