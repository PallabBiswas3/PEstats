from __future__ import annotations
import numpy as np
from scipy import stats as sp_stats
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class FeatureVector:
    """
    Container for all extracted features.
    Provides dict, array, and DataFrame export for ML pipelines.
    """
    features: dict = field(default_factory=dict)
    label:    str  = ""
    fs:       float = None

    def to_dict(self) -> dict:
        """Return flat dict — useful for building rows in a dataset."""
        return {**self.features, "label": self.label}

    def to_array(self) -> np.ndarray:
        """Return feature values as a 1D numpy array (for sklearn)."""
        return np.array(list(self.features.values()), dtype=float)

    def to_series(self) -> pd.Series:
        """Return as a pandas Series with feature names as index."""
        return pd.Series(self.features, name=self.label)

    def feature_names(self) -> list[str]:
        return list(self.features.keys())

    def summary(self) -> None:
        print(f"\nFeature vector — '{self.label}'  ({len(self.features)} features)")
        print("─" * 50)
        for name, val in self.features.items():
            print(f"  {name:<35s} {val:.6f}")
        print("─" * 50)


class FeatureMixin:
    """
    Extracts numerical features from signals for ML pipelines.

    Three categories:
      1. Time-domain  — shape, energy, complexity of the waveform
      2. Freq-domain  — spectral character of the signal
      3. Biomedical   — HRV metrics, peak analysis (ECG/EMG specific)

    All methods store results in self.feature_vector and return self.
    Call .auto_features() to run all three categories at once.
    Call .to_dataframe() on a batch for an sklearn-ready DataFrame.
    """

    # ── 1. Time-domain features ───────────────────────────────────────────

    def time_features(self) -> "FeatureMixin":
        """
        Extract time-domain features.

        Features extracted:
          mean, std, variance       — central tendency and spread
          skewness, kurtosis        — shape of the distribution
          rms                       — signal power (root mean square)
          peak_to_peak              — full amplitude range
          crest_factor              — peak / RMS ratio (impulsiveness)
          zero_crossing_rate        — how often signal crosses zero
                                      (high in noisy/complex signals)
          mean_absolute_value (MAV) — average rectified amplitude
                                      (standard EMG feature)
          waveform_length           — sum of absolute differences
                                      (measures signal complexity)
          hjorth_activity           — variance (signal power)
          hjorth_mobility           — std of 1st derivative / std of signal
                                      (mean frequency estimate)
          hjorth_complexity         — mobility of 1st deriv / mobility of signal
                                      (bandwidth measure)
          sample_entropy            — measure of signal regularity
                                      (low = regular, high = chaotic)
          permutation_entropy       — ordinal pattern complexity
        """
        x = self.data.flatten()

        # Basic stats
        mean   = float(np.mean(x))
        std    = float(np.std(x, ddof=1))
        rms    = float(np.sqrt(np.mean(x ** 2)))
        peak   = float(np.max(np.abs(x)))

        # Shape
        skew   = float(sp_stats.skew(x))
        kurt   = float(sp_stats.kurtosis(x))

        # Signal complexity
        zcr    = float(np.sum(np.diff(np.sign(x)) != 0) / len(x))
        mav    = float(np.mean(np.abs(x)))
        wl     = float(np.sum(np.abs(np.diff(x))))
        crest  = peak / (rms + 1e-12)
        p2p    = float(np.max(x) - np.min(x))

        # Hjorth parameters (used in EEG/EMG analysis)
        dx     = np.diff(x)
        ddx    = np.diff(dx)
        var_x  = np.var(x)
        var_dx = np.var(dx)
        var_ddx= np.var(ddx)

        activity   = float(var_x)
        mobility   = float(np.sqrt(var_dx / (var_x + 1e-12)))
        complexity = float(
            np.sqrt(var_ddx / (var_dx + 1e-12)) / (mobility + 1e-12)
        )

        # Sample entropy (regularity measure)
        se = self._sample_entropy(x, m=2, r=0.2 * std)

        # Permutation entropy
        pe = self._permutation_entropy(x, order=3)

        features = {
            "td_mean":               mean,
            "td_std":                std,
            "td_variance":           float(np.var(x, ddof=1)),
            "td_skewness":           skew,
            "td_kurtosis":           kurt,
            "td_rms":                rms,
            "td_peak_to_peak":       p2p,
            "td_crest_factor":       crest,
            "td_zero_crossing_rate": zcr,
            "td_mean_abs_value":     mav,
            "td_waveform_length":    wl,
            "td_hjorth_activity":    activity,
            "td_hjorth_mobility":    mobility,
            "td_hjorth_complexity":  complexity,
            "td_sample_entropy":     se,
            "td_permutation_entropy":pe,
        }

        self._merge_features(features)
        self._record("time_features()")
        return self

    # ── 2. Frequency-domain features ──────────────────────────────────────

    def freq_features(self) -> "FeatureMixin":
        """
        Extract frequency-domain features from the power spectrum.

        Features extracted:
          spectral_centroid     — "centre of gravity" of the spectrum.
                                  High = energy concentrated at high freq.
                                  Low  = energy at low frequencies.
          spectral_spread       — how wide the spectrum is around the centroid
          spectral_skewness     — asymmetry of the spectral distribution
          spectral_kurtosis     — peakedness of the spectral distribution
          spectral_flatness     — Wiener entropy. 1.0 = white noise (flat),
                                  near 0 = tonal signal (pure sine).
                                  Useful for distinguishing noise from signal.
          spectral_rolloff      — frequency below which 85% of power lies.
                                  Quick summary of dominant frequency range.
          spectral_entropy      — Shannon entropy of normalised power spectrum.
                                  High = complex/noisy signal.
          dominant_frequency    — frequency with maximum power
          band_power_*          — power in each EEG band (delta/theta/alpha/
                                  beta/gamma). Useful even for non-EEG signals
                                  as a coarse spectral decomposition.
          spectral_edge_*       — frequency below which X% of power lies
        """
        if self.fs is None:
            raise ValueError("fs required for frequency features.")

        x = self.data.flatten()
        n = len(x)

        # Compute power spectrum
        spectrum = rfft(x * np.hanning(n))
        freqs    = rfftfreq(n, d=1.0 / self.fs)
        power    = np.abs(spectrum) ** 2

        # Avoid division by zero
        total_power = power.sum() + 1e-12
        power_norm  = power / total_power

        # Spectral centroid
        centroid  = float(np.sum(freqs * power_norm))

        # Spectral spread (weighted std)
        spread = float(
            np.sqrt(np.sum(((freqs - centroid) ** 2) * power_norm))
        )

        # Spectral skewness and kurtosis
        spec_skew = float(
            np.sum(((freqs - centroid) ** 3) * power_norm) /
            (spread ** 3 + 1e-12)
        )
        spec_kurt = float(
            np.sum(((freqs - centroid) ** 4) * power_norm) /
            (spread ** 4 + 1e-12)
        )

        # Spectral flatness (geometric mean / arithmetic mean of power)
        log_power = np.log(power + 1e-12)
        flatness  = float(
            np.exp(np.mean(log_power)) / (np.mean(power) + 1e-12)
        )

        # Spectral rolloff (85% energy threshold)
        cumpower = np.cumsum(power)
        rolloff_idx = np.searchsorted(cumpower, 0.85 * cumpower[-1])
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        # Spectral entropy
        spec_entropy = float(-np.sum(power_norm * np.log2(power_norm + 1e-12)))

        # Dominant frequency
        dom_freq = float(freqs[np.argmax(power)])

        # Band powers (EEG bands — useful as spectral decomposition for any signal)
        def bp(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            return float(power[mask].sum() / total_power)

        # Spectral edge frequencies (SEF90, SEF95)
        sef90_idx = np.searchsorted(cumpower, 0.90 * cumpower[-1])
        sef95_idx = np.searchsorted(cumpower, 0.95 * cumpower[-1])

        features = {
            "fd_spectral_centroid":   centroid,
            "fd_spectral_spread":     spread,
            "fd_spectral_skewness":   spec_skew,
            "fd_spectral_kurtosis":   spec_kurt,
            "fd_spectral_flatness":   flatness,
            "fd_spectral_rolloff":    rolloff,
            "fd_spectral_entropy":    spec_entropy,
            "fd_dominant_frequency":  dom_freq,
            "fd_band_delta":          bp(0.5, 4),
            "fd_band_theta":          bp(4,   8),
            "fd_band_alpha":          bp(8,  13),
            "fd_band_beta":           bp(13, 30),
            "fd_band_gamma":          bp(30, 80),
            "fd_sef_90hz":            float(freqs[min(sef90_idx, len(freqs)-1)]),
            "fd_sef_95hz":            float(freqs[min(sef95_idx, len(freqs)-1)]),
        }

        self._merge_features(features)
        self._record("freq_features()")
        return self

    # ── 3. Biomedical / HRV features ──────────────────────────────────────

    def hrv_features(
        self,
        peak_distance: int = None,
        peak_height: float = None,
    ) -> "FeatureMixin":
        """
        Extract Heart Rate Variability (HRV) metrics from an ECG signal.

        HRV measures the variation in time between successive heartbeats
        (R-R intervals). It is one of the most clinically validated
        biomarkers for autonomic nervous system function.

        Standard HRV metrics extracted:

        Time-domain HRV:
          mean_rr       — average R-R interval in ms
          sdnn          — std of R-R intervals. Overall HRV.
                          < 50ms = unhealthy, 100ms+ = healthy
          rmssd         — RMS of successive RR differences.
                          Reflects parasympathetic activity.
          pnn50         — % of successive RR pairs differing > 50ms.
                          Standard clinical marker.
          mean_hr       — average heart rate (BPM)
          sdhr          — std of instantaneous heart rate

        Frequency-domain HRV:
          lf_power      — Low frequency (0.04–0.15 Hz) power.
                          Reflects sympathetic + parasympathetic activity.
          hf_power      — High frequency (0.15–0.4 Hz) power.
                          Reflects parasympathetic (vagal) activity.
          lf_hf_ratio   — LF/HF ratio. Sympathovagal balance.
                          > 2 suggests sympathetic dominance (stress).

        Geometric HRV:
          nn_triangular_index — total RR count / peak of histogram
          rr_range            — max - min RR interval

        Parameters
        ----------
        peak_distance : minimum samples between R-peaks.
                        Default: fs * 0.4 (minimum 40ms between peaks,
                        allows up to 150 BPM).
        peak_height   : minimum height threshold for R-peak detection.
                        Default: 0.5 × signal max.

        Example
        -------
        Data(ecg, fs=1000).highpass(0.5).notch(50).hrv_features()
        """
        if self.fs is None:
            raise ValueError("fs required for HRV analysis.")

        x = self.data.flatten()

        # Detect R-peaks
        min_dist   = peak_distance or int(self.fs * 0.4)
        min_height = peak_height   or (0.5 * np.max(x))

        peaks, props = find_peaks(x, distance=min_dist, height=min_height)

        if len(peaks) < 4:
            print(f"  Warning: only {len(peaks)} R-peaks detected. "
                  f"HRV requires at least 4. Check peak_height or apply "
                  f"bandpass(0.5, 40) + notch(50) before hrv_features().")
            self._merge_features({"hrv_peaks_detected": float(len(peaks))})
            self._record("hrv_features() — insufficient peaks")
            return self

        # R-R intervals in ms
        rr_ms   = np.diff(peaks) / self.fs * 1000.0

        # Time-domain HRV
        mean_rr = float(np.mean(rr_ms))
        sdnn    = float(np.std(rr_ms, ddof=1))
        rmssd   = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
        pnn50   = float(np.sum(np.abs(np.diff(rr_ms)) > 50) / len(rr_ms) * 100)
        hr_inst = 60000.0 / rr_ms         # instantaneous HR in BPM
        mean_hr = float(np.mean(hr_inst))
        sdhr    = float(np.std(hr_inst, ddof=1))
        rr_range= float(rr_ms.max() - rr_ms.min())

        # Triangular index (geometric HRV)
        hist, _ = np.histogram(rr_ms, bins=128)
        tri_idx = float(len(rr_ms) / (hist.max() + 1e-12))

        # Frequency-domain HRV (Welch PSD of RR series)
        hrv_features_dict = {
            "hrv_peaks_detected": float(len(peaks)),
            "hrv_mean_rr_ms":     mean_rr,
            "hrv_sdnn_ms":        sdnn,
            "hrv_rmssd_ms":       rmssd,
            "hrv_pnn50_pct":      pnn50,
            "hrv_mean_hr_bpm":    mean_hr,
            "hrv_sdhr_bpm":       sdhr,
            "hrv_rr_range_ms":    rr_range,
            "hrv_tri_index":      tri_idx,
        }

        # Only compute freq-domain HRV if we have enough beats
        if len(rr_ms) >= 20:
            try:
                from scipy.signal import welch
                # Interpolate RR to uniform grid at 4 Hz (standard)
                rr_times = np.cumsum(rr_ms) / 1000.0   # seconds
                t_uniform = np.arange(rr_times[0], rr_times[-1], 0.25)
                rr_interp = np.interp(t_uniform, rr_times, rr_ms)

                freqs_hrv, psd = welch(
                    rr_interp, fs=4.0, nperseg=min(256, len(rr_interp))
                )

                lf_mask = (freqs_hrv >= 0.04) & (freqs_hrv < 0.15)
                hf_mask = (freqs_hrv >= 0.15) & (freqs_hrv < 0.40)

                lf_power = float(np.trapz(psd[lf_mask], freqs_hrv[lf_mask]))
                hf_power = float(np.trapz(psd[hf_mask], freqs_hrv[hf_mask]))
                lf_hf    = float(lf_power / (hf_power + 1e-12))

                hrv_features_dict.update({
                    "hrv_lf_power":    lf_power,
                    "hrv_hf_power":    hf_power,
                    "hrv_lf_hf_ratio": lf_hf,
                })
            except Exception:
                pass   # silently skip freq HRV if signal is too short

        self._merge_features(hrv_features_dict)
        self._record(f"hrv_features(peaks={len(peaks)})")
        return self

    # ── 4. Peak features ──────────────────────────────────────────────────

    def peak_features(
        self,
        min_distance: int = None,
        min_height: float = None,
    ) -> "FeatureMixin":
        """
        Extract features from detected signal peaks.

        Useful for:
          - FTIR/Raman: peak count, dominant peak positions
          - EMG: burst detection (peak_rate as firing frequency)
          - EEG: spike counting in epileptic signals
          - General: any signal with repeated transient events

        Features extracted:
          peak_count          — number of peaks detected
          peak_rate_hz        — peaks per second
          mean_peak_height    — average amplitude at peaks
          std_peak_height     — variability of peak heights
          mean_peak_prominence — how much peaks stand out from baseline
          mean_peak_width     — average peak width in samples
          peak_regularity     — std of inter-peak intervals / mean
                                (0 = perfectly regular, >0.3 = irregular)
        """
        x = self.data.flatten()
        min_dist   = min_distance or max(int((self.fs or 100) * 0.1), 3)
        min_h      = min_height   or (np.mean(x) + 0.5 * np.std(x))

        peaks, props = find_peaks(
            x, distance=min_dist, height=min_h, prominence=0, width=0
        )
        prominences  = sp_stats.find_peaks(x, prominence=True)[1].get(
            "prominences", np.array([0.0])
        ) if len(peaks) > 0 else np.array([0.0])

        try:
            from scipy.signal import peak_prominences, peak_widths
            proms = peak_prominences(x, peaks)[0] if len(peaks) > 0 else np.array([0.0])
            widths= peak_widths(x, peaks, rel_height=0.5)[0] if len(peaks) > 0 else np.array([0.0])
        except Exception:
            proms  = np.array([0.0])
            widths = np.array([0.0])

        duration = len(x) / (self.fs or 1.0)
        ipi      = np.diff(peaks) if len(peaks) > 1 else np.array([0.0])

        features = {
            "pk_count":             float(len(peaks)),
            "pk_rate_hz":           float(len(peaks) / duration),
            "pk_mean_height":       float(x[peaks].mean()) if len(peaks) > 0 else 0.0,
            "pk_std_height":        float(x[peaks].std())  if len(peaks) > 0 else 0.0,
            "pk_mean_prominence":   float(proms.mean()),
            "pk_mean_width":        float(widths.mean()),
            "pk_regularity":        float(ipi.std() / (ipi.mean() + 1e-12)),
        }

        self._merge_features(features)
        self._record(f"peak_features(n_peaks={len(peaks)})")
        return self

    # ── 5. Auto extract all ────────────────────────────────────────────────

    def auto_features(
        self,
        include: list[str] = None,
    ) -> "FeatureMixin":
        """
        Extract ALL feature categories in one call.

        This is the recommended entry point for building ML datasets.
        Running after a full preprocessing + filter chain gives you a
        clean, ML-ready feature vector.

        Parameters
        ----------
        include : list of categories to run. Default = all.
                  Options: 'time', 'freq', 'peaks', 'hrv'
                  Example: include=['time', 'freq']  (skip HRV)

        Full example:
        -------------
        feature_row = (
            Data(ecg_signal, fs=1000, label="patient_001")
            .highpass(0.5)
            .notch(50)
            .bandpass(0.5, 40)
            .standardize()
            .auto_features()
            .feature_vector
            .to_dict()
        )
        """
        include = include or ["time", "freq", "peaks", "hrv"]

        if "time"  in include: self.time_features()
        if "freq"  in include: self.freq_features()
        if "peaks" in include: self.peak_features()
        if "hrv"   in include: self.hrv_features()

        self._record(f"auto_features(include={include})")
        return self

    # ── 6. DataFrame export for batch ML ──────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export the feature vector as a single-row DataFrame.

        Use this inside a batch loop:

            rows = []
            for signal, label in dataset:
                d = Data(signal, fs=1000, label=label)
                d.bandpass(0.5, 40).notch(50).auto_features()
                rows.append(d.to_dataframe())

            X = pd.concat(rows, ignore_index=True)
            y = X.pop("label")
            # X and y are now sklearn-ready
        """
        if not hasattr(self, "feature_vector"):
            raise RuntimeError(
                "No features extracted yet. "
                "Call .auto_features() or individual feature methods first."
            )
        return pd.DataFrame([self.feature_vector.to_dict()])

    # ── internal helpers ───────────────────────────────────────────────────

    def _merge_features(self, new_features: dict) -> None:
        """Add new features into self.feature_vector, creating it if needed."""
        if not hasattr(self, "feature_vector"):
            self.feature_vector = FeatureVector(label=self.label, fs=self.fs)
        self.feature_vector.features.update(new_features)
        self.feature_vector.label = self.label

    @staticmethod
    def _sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Sample entropy — measures signal irregularity.
        Uses a fast vectorised implementation.
        Low value = regular/predictable. High = complex/chaotic.
        """
        n = len(x)
        if n < 10:
            return 0.0
        # Use only up to 1000 samples for speed (representative subsample)
        x = x[:1000] if n > 1000 else x
        n = len(x)

        def count_matches(template_len):
            count = 0
            for i in range(n - template_len):
                template = x[i:i + template_len]
                for j in range(i + 1, n - template_len):
                    if np.max(np.abs(x[j:j + template_len] - template)) <= r:
                        count += 1
            return count

        A = count_matches(m + 1)
        B = count_matches(m)
        if B == 0:
            return 0.0
        return float(-np.log((A + 1e-12) / (B + 1e-12)))

    @staticmethod
    def _permutation_entropy(x: np.ndarray, order: int = 3) -> float:
        """
        Permutation entropy — ordinal pattern complexity.
        Faster and more robust than sample entropy for long signals.
        """
        n = len(x)
        if n < order + 1:
            return 0.0
        permutations = {}
        for i in range(n - order):
            pattern = tuple(np.argsort(x[i:i + order]))
            permutations[pattern] = permutations.get(pattern, 0) + 1
        total = sum(permutations.values())
        probs = np.array(list(permutations.values())) / total
        return float(-np.sum(probs * np.log2(probs + 1e-12)))