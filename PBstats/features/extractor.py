from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from PBstats.core.steps import step
from PBstats.core.utils import safe_print
from PBstats.features import functions as F


@dataclass
class FeatureVector:
    """
    Container for extracted features, with dict / array / pandas export
    for ML pipelines.
    """
    features: dict = field(default_factory=dict)
    label: str = ""
    fs: float = None

    def to_dict(self) -> dict:
        """Flat dict — one row of a dataset."""
        return {**self.features, "label": self.label}

    def to_array(self) -> np.ndarray:
        """Feature values as a 1D float array (for sklearn)."""
        return np.array(list(self.features.values()), dtype=float)

    def to_series(self) -> pd.Series:
        return pd.Series(self.features, name=self.label, dtype=float)

    def feature_names(self) -> list[str]:
        return list(self.features.keys())

    def __len__(self) -> int:
        return len(self.features)

    def summary(self) -> None:
        safe_print(f"\nFeature vector — '{self.label}'  ({len(self.features)} features)")
        safe_print("─" * 50)
        for name, val in self.features.items():
            safe_print(f"  {name:<35s} {val:.6g}")
        safe_print("─" * 50)


class FeatureMixin:
    """
    Numerical feature extraction for machine learning.

    Categories (each stored into ``self.feature_vector``):
      time      — amplitude, shape, Hjorth parameters           (td_*)
      freq      — Welch-PSD spectral shape, band powers, edges   (fd_*)
      nonlinear — sample/permutation entropy, Higuchi FD, DFA α  (nl_*)
      peaks     — event count, rate, prominence, regularity      (pk_*)
      hrv       — heart-rate variability from ECG R-peaks        (hrv_*)

    Multichannel data: features are computed per channel and prefixed with
    the channel name, e.g. ``Fz__td_rms``.
    """

    # ── per-category entry points ─────────────────────────────────────────

    @step
    def time_features(self) -> "FeatureMixin":
        """Time-domain features (td_*): mean, std, skewness, kurtosis, RMS,
        peak-to-peak, crest factor, zero-crossing rate, mean absolute value,
        waveform length, Hjorth activity / mobility / complexity."""
        self._extract_per_channel(F.time_domain, "time_features")
        return self

    @step
    def freq_features(self) -> "FeatureMixin":
        """Frequency-domain features (fd_*) from a Welch PSD: spectral
        centroid, spread, skewness, kurtosis, flatness (1 = white noise,
        ~0 = tonal), rolloff, normalised entropy, dominant frequency,
        SEF90/95 and relative EEG band powers."""
        self._require_fs("freq_features")
        self._extract_per_channel(lambda c: F.frequency_domain(c, self.fs), "freq_features")
        return self

    @step
    def nonlinear_features(self) -> "FeatureMixin":
        """Complexity features (nl_*): sample entropy, normalised permutation
        entropy, Higuchi fractal dimension, DFA scaling exponent α."""
        self._extract_per_channel(F.nonlinear, "nonlinear_features")
        return self

    @step
    def peak_features(
        self, min_distance: int | None = None, min_height: float | None = None,
    ) -> "FeatureMixin":
        """Peak/event features (pk_*): count, rate (Hz), mean/std height,
        mean prominence, mean width (s), regularity (CV of inter-peak
        intervals; 0 = perfectly regular)."""
        self._extract_per_channel(
            lambda c: F.peaks(c, self.fs, min_distance, min_height), "peak_features",
        )
        return self

    @step
    def hrv_features(
        self,
        peak_distance: int | None = None,
        peak_height: float | None = None,
        rr_filter: float | None = 0.3,
    ) -> "FeatureMixin":
        """
        Heart-rate variability (hrv_*) from an ECG: R-peaks are detected,
        R-R intervals derived and artefact-filtered, and SDNN, RMSSD, pNN50,
        HR, triangular index, Poincaré SD1/SD2 and LF/HF power computed.
        Missing metrics are NaN. R-peak indices are kept in ``self.r_peaks``.

        rr_filter : reject R-R intervals deviating more than this fraction from
                    the local median (missed/extra beats, ectopics).
                    None keeps every interval.

        Clean the ECG first, e.g.:
            Data(ecg, fs=500).bandpass(0.5, 40).notch(50).hrv_features()
        """
        self._require_fs("hrv_features")

        def _hrv(c):
            r = F.detect_r_peaks(c, self.fs, peak_distance, peak_height)
            if len(r) < 4:
                warnings.warn(
                    f"Only {len(r)} R-peaks detected; HRV needs >= 4. Check peak_height "
                    f"or clean the ECG (bandpass(0.5, 40), notch(50)) first.",
                    stacklevel=4,
                )
            self.r_peaks = r
            return F.hrv(np.diff(r) / self.fs * 1000.0, rr_filter=rr_filter)

        self._extract_per_channel(_hrv, "hrv_features")
        return self

    @step
    def auto_features(self, include: list[str] | None = None) -> "FeatureMixin":
        """
        Extract several categories in one call — the recommended entry point
        for building ML datasets.

        include : subset of ['time', 'freq', 'nonlinear', 'peaks', 'hrv'].
                  Default: all but 'hrv' (which only makes sense for ECG).

        Example
        -------
        row = (Data(ecg, fs=500, label="patient_001")
               .bandpass(0.5, 40).notch(50)
               .auto_features(include=["time", "freq", "hrv"])
               .feature_vector.to_dict())
        """
        include = list(include) if include is not None else list(F.DEFAULT_CATEGORIES)
        unknown = set(include) - set(F.CATEGORIES)
        if unknown:
            raise ValueError(f"Unknown categories {sorted(unknown)}. Choose from {F.CATEGORIES}")
        dispatch = {
            "time": self.time_features,
            "freq": self.freq_features,
            "nonlinear": self.nonlinear_features,
            "peaks": self.peak_features,
            "hrv": self.hrv_features,
        }
        for cat in include:
            dispatch[cat]()
        self._record(f"auto_features(include={include})")
        return self

    # ── windowed extraction ───────────────────────────────────────────────

    def windowed_features(
        self,
        window: float,
        step: float | None = None,
        include: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Slide a window over the signal and extract features from each one —
        turns one long recording into a feature *time series*, the standard
        input for activity recognition, sleep staging, fault detection,
        seizure detection, etc.

        Parameters
        ----------
        window : window length in seconds (samples if fs is not set)
        step   : hop between windows (default: window, i.e. no overlap)
        include: feature categories (see auto_features)

        Returns
        -------
        DataFrame with one row per window: ``t_start``, ``t_end`` and features.
        """
        from PBstats.core.utils import require_1d
        x = require_1d(self.data, "windowed_features")
        scale = self.fs or 1.0
        win = int(round(window * scale))
        hop = int(round((step or window) * scale))
        if win < 4 or hop < 1:
            raise ValueError("window too short (need >= 4 samples) or step < 1 sample")
        if win > len(x):
            raise ValueError(f"window ({win} samples) longer than signal ({len(x)})")

        rows = []
        for start in range(0, len(x) - win + 1, hop):
            feats = F.extract(x[start:start + win], self.fs, include)
            rows.append({"t_start": start / scale, "t_end": (start + win) / scale, **feats})
        df = pd.DataFrame(rows)
        df.insert(0, "label", self.label)
        return df

    # ── export ────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """
        The feature vector as a single-row DataFrame. Collect rows in a loop
        and ``pd.concat`` them — or use ``PBstats.extract_features`` /
        ``PBstats.SignalFeatureExtractor`` to do it for you.
        """
        if not hasattr(self, "feature_vector"):
            raise RuntimeError(
                "No features extracted yet. Call .auto_features() or a *_features() method first."
            )
        return pd.DataFrame([self.feature_vector.to_dict()])

    # ── internals ─────────────────────────────────────────────────────────

    def _extract_per_channel(self, fn: Callable[[np.ndarray], dict], name: str) -> None:
        x = self.data
        if np.isnan(x).any():
            raise ValueError(f"{name}(): data contains NaN — call remove_missing() first.")
        if x.ndim == 1:
            feats = fn(x)
        else:
            names = self.channel_names
            feats = {}
            for c in range(x.shape[1]):
                feats.update({f"{names[c]}__{k}": v for k, v in fn(x[:, c]).items()})
        self._merge_features(feats)
        self._record(f"{name}()")

    def _merge_features(self, new_features: dict) -> None:
        if not hasattr(self, "feature_vector"):
            self.feature_vector = FeatureVector(label=self.label, fs=self.fs)
        self.feature_vector.features.update(new_features)
        self.feature_vector.label = self.label
