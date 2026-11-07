from __future__ import annotations

import copy as _copy
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import pandas as pd

from PBstats.diagnostics.quality import QualityMixin
from PBstats.features.extractor import FeatureMixin
from PBstats.preprocessing.cleaning import CleaningMixin
from PBstats.preprocessing.filters import FilterMixin
from PBstats.preprocessing.imputation import ImputationMixin
from PBstats.preprocessing.normalization import NormalizationMixin
from PBstats.preprocessing.outliers import OutlierMixin
from PBstats.statistics.correlation import CorrelationMixin
from PBstats.statistics.descriptive import DescriptiveMixin
from PBstats.statistics.hypothesis import HypothesisMixin
from PBstats.statistics.regression import RegressionMixin
from PBstats.transforms.fft import FFTMixin
from PBstats.transforms.hilbert import HilbertMixin
from PBstats.transforms.stft import STFTMixin
from PBstats.transforms.wavelet import WaveletMixin
from PBstats.visualization.spectrum import VisualizationMixin

if TYPE_CHECKING:
    from PBstats.core.pipeline import Pipeline

ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame, list]

# Attributes holding analysis results — cleared by reset().
_RESULT_ATTRS = ("autocorr", "autocorr_n", "feature_vector", "quality_report",
                 "quality_report_after", "r_peaks")


class Data(
    # Preprocessing
    CleaningMixin, NormalizationMixin, OutlierMixin, ImputationMixin, FilterMixin,
    # Transforms
    FFTMixin, HilbertMixin, WaveletMixin, STFTMixin,
    # Statistics
    DescriptiveMixin, HypothesisMixin, RegressionMixin, CorrelationMixin,
    # Features, diagnostics
    FeatureMixin, QualityMixin,
    # Output
    VisualizationMixin,
):
    """
    A signal plus everything done to it.

    Parameters
    ----------
    data          : 1D array (one channel) or 2D array ``(n_samples, n_channels)``;
                    lists, pandas Series and DataFrames (columns = channels)
                    are accepted.
    fs            : sampling frequency in Hz (needed for filters / spectra)
    label         : name used in plots, reports and feature tables
    channel_names : names for 2D data (default: DataFrame columns or ch0, ch1, …)

    Every processing method mutates ``self.data`` and returns ``self``, so
    calls chain. The original input is kept, so ``reset()`` can undo
    everything, and every call is recorded so ``to_pipeline()`` can replay it.
    """

    def __init__(
        self,
        data: ArrayLike,
        fs: Optional[float] = None,
        label: str = "",
        channel_names: Optional[Sequence[str]] = None,
    ):
        if fs is not None and not fs > 0:
            raise ValueError(f"fs must be positive, got {fs}")
        if channel_names is None and isinstance(data, pd.DataFrame):
            channel_names = [str(c) for c in data.columns]

        self._raw = self._coerce(data)
        self.data = self._raw.copy()
        self._raw_fs = float(fs) if fs is not None else None
        self.fs = self._raw_fs
        self.label = label
        self._channel_names = list(channel_names) if channel_names is not None else None
        if self._channel_names is not None and self._raw.ndim == 2 \
                and len(self._channel_names) != self._raw.shape[1]:
            raise ValueError(
                f"{len(self._channel_names)} channel names for {self._raw.shape[1]} channels"
            )
        self._log: list[str] = []
        self._steps: list[tuple[str, dict]] = []

    # ── construction helpers ──────────────────────────────────────────────

    @staticmethod
    def _coerce(data: ArrayLike) -> np.ndarray:
        if isinstance(data, (pd.DataFrame, pd.Series)):
            arr = data.to_numpy(dtype=float)
        else:
            arr = np.asarray(data, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim == 0 or arr.ndim > 2:
            raise ValueError(
                f"Data must be 1D (one channel) or 2D (n_samples, n_channels); got shape {arr.shape}"
            )
        if arr.shape[0] == 0:
            raise ValueError("Data is empty")
        return arr

    def _require_fs(self, method_name: str) -> None:
        if self.fs is None:
            raise ValueError(
                f"{method_name}() requires a sampling frequency. "
                f"Set it when creating Data: Data(signal, fs=1000)"
            )

    def _record(self, s: str) -> None:
        self._log.append(s)

    # ── state ─────────────────────────────────────────────────────────────

    def reset(self) -> "Data":
        """Restore the original data and fs, and clear history and results."""
        self.data = self._raw.copy()
        self.fs = self._raw_fs
        self._log.clear()
        self._steps.clear()
        for attr in list(vars(self)):
            if attr.endswith("_result") or attr in _RESULT_ATTRS:
                delattr(self, attr)
        return self

    def copy(self) -> "Data":
        """Independent deep copy (data, history and results)."""
        return _copy.deepcopy(self)

    def history(self) -> list[str]:
        """Human-readable log of operations."""
        return self._log.copy()

    @property
    def steps(self) -> list[tuple[str, dict]]:
        """Structured record of every method called: [(name, kwargs), ...]."""
        return [(n, dict(kw)) for n, kw in self._steps]

    def to_pipeline(self, name: str = "session") -> "Pipeline":
        """
        Turn everything done to this object into a reusable ``Pipeline``.

        Explore interactively on one recording, then replay the exact same
        processing on a whole dataset:

            d = Data(x, fs=500).bandpass(0.5, 40).notch(50).standardize()
            pipe = d.to_pipeline("ecg_clean")
            pipe.save("ecg_clean.json")
            cleaned = pipe.run_batch([Data(s, fs=500) for s in dataset])
        """
        from PBstats.core.pipeline import Pipeline
        pipe = Pipeline(name)
        for step_name, kwargs in self._steps:
            pipe.add(step_name, **kwargs)
        return pipe

    # ── shape / channels ──────────────────────────────────────────────────

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        return 1 if self.data.ndim == 1 else int(self.data.shape[1])

    @property
    def channel_names(self) -> list[str]:
        if self._channel_names is not None and len(self._channel_names) == self.n_channels:
            return list(self._channel_names)
        return [f"ch{i}" for i in range(self.n_channels)]

    @property
    def duration(self) -> Optional[float]:
        """Length in seconds (None without fs)."""
        return self.n_samples / self.fs if self.fs else None

    @property
    def times(self) -> np.ndarray:
        """Time axis in seconds (sample index without fs)."""
        return np.arange(self.n_samples) / (self.fs or 1.0)

    def channel(self, key: Union[int, str]) -> "Data":
        """A new single-channel Data object for one channel (by index or name)."""
        if self.data.ndim == 1:
            if key in (0, self.channel_names[0]):
                return Data(self.data.copy(), fs=self.fs, label=self.label)
            raise IndexError(f"Single-channel data has no channel {key!r}")
        names = self.channel_names
        idx = names.index(key) if isinstance(key, str) else int(key)
        label = f"{self.label}[{names[idx]}]" if self.label else names[idx]
        return Data(self.data[:, idx].copy(), fs=self.fs, label=label)

    # ── export ────────────────────────────────────────────────────────────

    def to_numpy(self) -> np.ndarray:
        return self.data.copy()

    def to_series(self) -> pd.Series:
        return pd.Series(self.data.flatten(), name=self.label)

    def to_frame(self) -> pd.DataFrame:
        """Current data as a DataFrame indexed by time, one column per channel."""
        cols = self.channel_names if self.data.ndim == 2 else [self.label or "signal"]
        arr = self.data if self.data.ndim == 2 else self.data[:, None]
        idx = pd.Index(self.times, name="time_s" if self.fs else "sample")
        return pd.DataFrame(arr, index=idx, columns=cols)

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        shape = self.data.shape
        fs = f"{self.fs:g} Hz" if self.fs else None
        return f"Data('{self.label}', shape={shape}, fs={fs}, steps={len(self._log)})"
