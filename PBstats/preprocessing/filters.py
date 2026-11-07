from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import (
    butter,
    filtfilt,
    firwin,
    iirnotch,
    lfilter,
    resample_poly,
    savgol_filter,
    sosfilt,
    sosfiltfilt,
)

from PBstats.core.steps import step
from PBstats.core.utils import check_length, per_channel, safe_print


@dataclass
class FilterResult:
    """Metadata about the last filter applied."""
    filter_type: str
    cutoff: object          # float, tuple or list
    order: int
    fs: float
    zero_phase: bool = True
    label: str = ""

    def summary(self) -> None:
        safe_print(f"\nFilter applied: {self.filter_type}")
        safe_print(f"  Cutoff  : {self.cutoff} Hz")
        safe_print(f"  Order   : {self.order}")
        safe_print(f"  fs      : {self.fs} Hz")
        safe_print(f"  Method  : {'filtfilt (zero-phase)' if self.zero_phase else 'causal'}")


class FilterMixin:
    """
    Frequency-selective filtering. Zero-phase (forward-backward) by default,
    so peaks stay exactly where they were in time — critical for ECG/EEG
    timing analysis. Multichannel data is filtered along time (axis 0).

    Every method returns self:
        Data(ecg, fs=1000).notch(50).bandpass(0.5, 40).hilbert()
    """

    # ── internals ─────────────────────────────────────────────────────────

    def _nyquist(self, method: str) -> float:
        self._require_fs(method)
        return self.fs / 2.0

    def _check_cutoff(self, f: float, nyq: float, method: str) -> None:
        if not 0 < f < nyq:
            raise ValueError(
                f"{method}: cutoff {f} Hz must be within (0, {nyq}) Hz (Nyquist = fs/2)"
            )

    def _apply_sos(self, sos: np.ndarray, zero_phase: bool, method: str) -> None:
        x = self.data
        if zero_phase:
            # sosfiltfilt's default padding length
            padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
            check_length(x, padlen + 1, method)
            self.data = sosfiltfilt(sos, x, axis=0)
        else:
            self.data = sosfilt(sos, x, axis=0)

    def _butter(self, kind, cutoff, btype, order, zero_phase, record):
        nyq = self._nyquist(kind)
        cutoffs = cutoff if isinstance(cutoff, (tuple, list)) else (cutoff,)
        for c in cutoffs:
            self._check_cutoff(c, nyq, kind)
        if len(cutoffs) == 2 and cutoffs[0] >= cutoffs[1]:
            raise ValueError(f"{kind}: low ({cutoffs[0]}) must be less than high ({cutoffs[1]})")

        sos = butter(order, cutoff, btype=btype, fs=self.fs, output="sos")
        self._apply_sos(sos, zero_phase, kind)
        self.filter_result = FilterResult(
            filter_type=f"Butterworth {kind}", cutoff=cutoff, order=order,
            fs=self.fs, zero_phase=zero_phase, label=self.label,
        )
        self._record(record)
        return self

    # ── Butterworth family ────────────────────────────────────────────────

    @step
    def lowpass(self, cutoff: float, order: int = 4, zero_phase: bool = True) -> "FilterMixin":
        """
        Remove frequencies ABOVE cutoff (EMG noise in ECG ~40 Hz, smoothing,
        anti-aliasing). order: steepness, 4 is standard; avoid > 8.

        Example: Data(eeg, fs=256).lowpass(40)
        """
        return self._butter("lowpass", cutoff, "low", order, zero_phase,
                            f"lowpass(cutoff={cutoff}Hz, order={order})")

    @step
    def highpass(self, cutoff: float, order: int = 4, zero_phase: bool = True) -> "FilterMixin":
        """
        Remove frequencies BELOW cutoff (baseline wander ~0.5 Hz, DC drift).

        Example: Data(ecg, fs=1000).highpass(0.5)
        """
        return self._butter("highpass", cutoff, "high", order, zero_phase,
                            f"highpass(cutoff={cutoff}Hz, order={order})")

    @step
    def bandpass(
        self, low: float, high: float, order: int = 4, zero_phase: bool = True,
    ) -> "FilterMixin":
        """
        Keep only [low, high] Hz.

        EEG bands: delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–80.
        ECG: bandpass(0.5, 40). EMG: bandpass(20, 450).

        Example: Data(eeg, fs=256).bandpass(8, 13)   # alpha
        """
        return self._butter("bandpass", (low, high), "bandpass", order, zero_phase,
                            f"bandpass(low={low}Hz, high={high}Hz, order={order})")

    @step
    def bandstop(
        self, low: float, high: float, order: int = 4, zero_phase: bool = True,
    ) -> "FilterMixin":
        """
        Remove the band [low, high] Hz. For a single sharp frequency use notch().

        Example: Data(sig, fs=1000).bandstop(45, 55)
        """
        return self._butter("bandstop", (low, high), "bandstop", order, zero_phase,
                            f"bandstop(low={low}Hz, high={high}Hz, order={order})")

    # ── Notch ─────────────────────────────────────────────────────────────

    @step
    def notch(
        self,
        freq: float,
        quality: float = 30.0,
        harmonics: bool = False,
        zero_phase: bool = True,
    ) -> "FilterMixin":
        """
        Surgically remove one frequency — the standard fix for power-line
        interference (50 Hz: Europe/Asia/Africa/India; 60 Hz: Americas).

        quality   : Q factor; higher = narrower notch. 30 is the clinical
                    default; ~10 for drifting interference.
        harmonics : also notch 2f, 3f, ... below Nyquist.

        Example: Data(ecg, fs=1000).notch(50, harmonics=True)
        """
        nyq = self._nyquist("notch")
        self._check_cutoff(freq, nyq, "notch")

        freqs = [freq]
        if harmonics:
            k = 2
            while freq * k < nyq:
                freqs.append(freq * k)
                k += 1

        for f in freqs:
            b, a = iirnotch(f, quality, fs=self.fs)
            if zero_phase:
                check_length(self.data, 3 * max(len(a), len(b)) + 1, "notch")
                self.data = filtfilt(b, a, self.data, axis=0)
            else:
                self.data = lfilter(b, a, self.data, axis=0)

        self.filter_result = FilterResult(
            filter_type="Notch (IIR)", cutoff=freqs, order=2,
            fs=self.fs, zero_phase=zero_phase, label=self.label,
        )
        self._record(f"notch(freq={freq}Hz, Q={quality}, harmonics={harmonics})")
        return self

    # ── FIR ───────────────────────────────────────────────────────────────

    @step
    def fir_filter(
        self,
        cutoff,
        filter_type: str = "lowpass",
        numtaps: int = 101,
        window: str = "hamming",
    ) -> "FilterMixin":
        """
        Linear-phase FIR filter (window method), applied zero-phase.
        Prefer it over Butterworth when waveform shape must be preserved
        (ERPs, spike sorting) or guaranteed stability matters.

        cutoff      : float or [low, high] Hz
        filter_type : 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
        numtaps     : filter length (forced odd)

        Example: Data(erp, fs=1000).fir_filter([1, 40], filter_type="bandpass")
        """
        nyq = self._nyquist("fir_filter")
        valid = ("lowpass", "highpass", "bandpass", "bandstop")
        if filter_type not in valid:
            raise ValueError(f"filter_type must be one of {valid}")
        cutoffs = [cutoff] if np.isscalar(cutoff) else list(cutoff)
        for c in cutoffs:
            self._check_cutoff(c, nyq, "fir_filter")
        if numtaps % 2 == 0:
            numtaps += 1  # odd length => Type I, valid for every response type

        taps = firwin(numtaps, cutoffs if len(cutoffs) > 1 else cutoffs[0],
                      window=window, pass_zero=filter_type in ("lowpass", "bandstop"),
                      fs=self.fs)
        check_length(self.data, 3 * numtaps + 1, "fir_filter")
        self.data = filtfilt(taps, [1.0], self.data, axis=0)

        self.filter_result = FilterResult(
            filter_type=f"FIR {filter_type}", cutoff=cutoff, order=numtaps,
            fs=self.fs, zero_phase=True, label=self.label,
        )
        self._record(f"fir_filter(cutoff={cutoff}, type={filter_type}, taps={numtaps})")
        return self

    # ── Smoothing ─────────────────────────────────────────────────────────

    @step
    def savgol(self, window_length: int = 11, polyorder: int = 3) -> "FilterMixin":
        """
        Savitzky-Golay smoothing: fits local polynomials, preserving peak
        height and position far better than a lowpass. Standard for
        FTIR/Raman spectra and QRS-preserving ECG smoothing.

        Example: Data(ftir).savgol(window_length=15, polyorder=3)
        """
        if window_length % 2 == 0:
            window_length += 1
        if window_length <= polyorder:
            raise ValueError("window_length must be greater than polyorder")
        check_length(self.data, window_length, "savgol")
        self.data = savgol_filter(self.data, window_length, polyorder, axis=0)
        self._record(f"savgol(window={window_length}, poly={polyorder})")
        return self

    @step
    def median_filter(self, kernel: int = 5) -> "FilterMixin":
        """
        Rolling median — removes isolated spikes / impulse noise while
        keeping edges sharp (a linear filter would smear them).
        """
        from PBstats.preprocessing.outliers import rolling_median
        self.data = per_channel(lambda c: rolling_median(c, kernel), self.data)
        self._record(f"median_filter(kernel={kernel})")
        return self

    # ── Resampling ────────────────────────────────────────────────────────

    @step
    def resample(self, target_fs: float) -> "FilterMixin":
        """
        Resample to a new rate with polyphase filtering (built-in
        anti-aliasing). Handles non-integer rates, e.g. 250.5 → 128 Hz.

        Example: Data(eeg, fs=2048).resample(256)
        """
        self._require_fs("resample")
        if target_fs <= 0:
            raise ValueError("target_fs must be positive")
        ratio = Fraction(float(target_fs) / float(self.fs)).limit_denominator(1000)
        self.data = resample_poly(self.data, ratio.numerator, ratio.denominator, axis=0)

        old_fs = self.fs
        self.fs = float(target_fs)
        self._record(f"resample({old_fs}Hz → {target_fs}Hz)")
        return self

    # ── Visual check ──────────────────────────────────────────────────────

    def plot_filter_response(
        self,
        cutoff,
        filter_type: str = "bandpass",
        order: int = 4,
        quality: float = 30.0,
        save_path: str | None = None,
    ) -> "FilterMixin":
        """
        Plot magnitude and phase response of a filter BEFORE applying it.

        filter_type : 'lowpass' | 'highpass' | 'bandpass' | 'bandstop' | 'notch'
        """
        self._require_fs("plot_filter_response")
        import matplotlib.pyplot as plt
        from scipy.signal import freqz, sosfreqz

        from PBstats.visualization.base import COLORS, save_or_show, style

        if filter_type == "notch":
            b, a = iirnotch(cutoff, quality, fs=self.fs)
            w, h = freqz(b, a, worN=4096, fs=self.fs)
        else:
            btype = {"lowpass": "low", "highpass": "high",
                     "bandpass": "bandpass", "bandstop": "bandstop"}.get(filter_type)
            if btype is None:
                raise ValueError(f"Unknown filter_type '{filter_type}'")
            sos = butter(order, cutoff, btype=btype, fs=self.fs, output="sos")
            w, h = sosfreqz(sos, worN=4096, fs=self.fs)

        with style():
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            fig.suptitle(f"Filter response — {filter_type}  cutoff={cutoff} Hz",
                         fontsize=12, fontweight="bold")
            axes[0].plot(w, 20 * np.log10(np.abs(h) + 1e-12), color=COLORS["primary"])
            axes[0].axhline(-3, color=COLORS["accent"], ls="--", lw=1, label="-3 dB")
            axes[0].axhline(-40, color=COLORS["neutral"], ls=":", lw=0.8, label="-40 dB")
            axes[0].set_ylabel("Magnitude (dB)")
            axes[0].set_ylim(bottom=max(axes[0].get_ylim()[0], -120))
            axes[0].legend(fontsize=9, frameon=False)
            axes[1].plot(w, np.unwrap(np.angle(h)), color=COLORS["secondary"])
            axes[1].set_xlabel("Frequency (Hz)")
            axes[1].set_ylabel("Phase (rad)")
            note = "zero-phase filtering cancels this phase shift" if filter_type != "notch" else ""
            axes[1].set_title(f"Phase response  {note}", fontsize=10)
            save_or_show(fig, save_path)
        return self
