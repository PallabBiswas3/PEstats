from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Union, Optional

from PBstats.preprocessing.cleaning      import CleaningMixin
from PBstats.preprocessing.normalization import NormalizationMixin
from PBstats.preprocessing.outliers      import OutlierMixin
from PBstats.preprocessing.imputation    import ImputationMixin
from PBstats.preprocessing.filters       import FilterMixin

from PBstats.transforms.fft              import FFTMixin
from PBstats.transforms.hilbert          import HilbertMixin
from PBstats.transforms.wavelet          import WaveletMixin
from PBstats.transforms.stft             import STFTMixin

from PBstats.statistics.descriptive      import DescriptiveMixin
from PBstats.statistics.hypothesis       import HypothesisMixin
from PBstats.statistics.regression       import RegressionMixin
from PBstats.statistics.correlation      import CorrelationMixin

from PBstats.visualization.spectrum      import VisualizationMixin
from PBstats.features.extractor          import FeatureMixin


ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame, list]

class Data(
    # Preprocessing
    CleaningMixin, NormalizationMixin, OutlierMixin,
    ImputationMixin, FilterMixin,

    # Transforms
    FFTMixin, HilbertMixin, WaveletMixin, STFTMixin,

    # Statistics
    DescriptiveMixin, HypothesisMixin,
    RegressionMixin, CorrelationMixin,

    # Features
    FeatureMixin,

    # Output
    VisualizationMixin, 
):
    def __init__(
        self,
        data: ArrayLike,
        fs: Optional[float] = None,
        label: str = "",
    ):
        self._raw  = self._coerce(data)
        self.data  = self._raw.copy()
        self.fs    = fs
        self.label = label
        self._log: list[str] = []

    def _coerce(self, data: ArrayLike) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            return data.to_numpy()

        if isinstance(data, pd.Series):
            return data.to_numpy()

        return np.asarray(data, dtype=float)

    def _record(self, s: str) -> None:
        self._log.append(s)

    def reset(self) -> "Data":
        self.data = self._raw.copy()
        self._log.clear()
        return self

    def history(self) -> list[str]:
        return self._log.copy()

    def to_numpy(self) -> np.ndarray:
        return self.data.copy()

    def to_series(self) -> pd.Series:
        return pd.Series(self.data.flatten(), name=self.label)

    def __repr__(self) -> str:
        return (
            f"Data(shape={self.data.shape}, "
            f"fs={self.fs}, "
            f"steps={len(self._log)})"
        )