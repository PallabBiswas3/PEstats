from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

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
    rms: float          # root mean square — essential for signal energy
    snr: float          # signal-to-noise ratio estimate
    n: int              # sample count
    label: str = ""

    def summary(self) -> None:
        print(f"\nDescriptive Stats — '{self.label}'")
        print("─" * 40)
        print(f"  N          : {self.n}")
        print(f"  Mean       : {self.mean:.6f}")
        print(f"  Median     : {self.median:.6f}")
        print(f"  Std        : {self.std:.6f}")
        print(f"  Skewness   : {self.skewness:.6f}")
        print(f"  Kurtosis   : {self.kurtosis:.6f}")
        print(f"  Min / Max  : {self.minimum:.4f} / {self.maximum:.4f}")
        print(f"  Q25 / Q75  : {self.q25:.4f} / {self.q75:.4f}")
        print(f"  IQR        : {self.iqr:.6f}")
        print(f"  RMS        : {self.rms:.6f}")
        print(f"  SNR (est.) : {self.snr:.2f} dB")
        print("─" * 40)

    def to_dict(self) -> dict:
        return {
            "label": self.label, "n": self.n,
            "mean": self.mean, "median": self.median,
            "std": self.std, "variance": self.variance,
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

    def summary(self) -> None:
        verdict = "REJECT H0" if self.reject_null else "FAIL TO REJECT H0"
        print(f"\nHypothesis Test — {self.test}")
        print("─" * 40)
        print(f"  Statistic  : {self.statistic:.6f}")
        print(f"  p-value    : {self.p_value:.6f}")
        print(f"  Alpha      : {self.alpha}")
        print(f"  Decision   : {verdict}")
        print(f"  Note       : {self.interpretation}")
        print("─" * 40)


@dataclass
class CorrelationResult:
    method: str
    matrix: np.ndarray            # correlation matrix (n_signals × n_signals)
    labels: list[str]
    pairwise: list[dict]          # flat list of (label_a, label_b, r, p)

    def strongest(self, n: int = 3) -> list[dict]:
        """Return the n strongest correlations (by |r|), excluding self-pairs."""
        ranked = sorted(
            [p for p in self.pairwise if p["label_a"] != p["label_b"]],
            key=lambda x: abs(x["r"]),
            reverse=True,
        )
        return ranked[:n]

    def summary(self) -> None:
        print(f"\nCorrelation ({self.method})")
        print("─" * 40)
        for pair in self.strongest(5):
            print(f"  {pair['label_a']:15s} × {pair['label_b']:15s} "
                  f"r={pair['r']:+.4f}  p={pair['p']:.4f}")
        print("─" * 40)