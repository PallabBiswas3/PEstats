from __future__ import annotations
import numpy as np
from scipy.signal import (
    butter, iirnotch, iirpeak,
    filtfilt, sosfiltfilt,
    resample_poly, firwin,
    butter as _butter,
)
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class FilterResult:
    """Stores metadata about the last filter applied."""
    filter_type:  str
    cutoff:       object          # float or tuple
    order:        int
    fs:           float
    zero_phase:   bool = True
    label:        str  = ""

    def summary(self) -> None:
        print(f"\nFilter applied: {self.filter_type}")
        print(f"  Cutoff  : {self.cutoff} Hz")
        print(f"  Order   : {self.order}")
        print(f"  fs      : {self.fs} Hz")
        print(f"  Method  : {'filtfilt (zero-phase)' if self.zero_phase else 'lfilter (causal)'}")


class FilterMixin:
    """
    Adds surgical filtering methods to the Data class.

    All filters use filtfilt (zero-phase) by default.
    filtfilt applies the filter twice — forward and backward —
    which cancels phase distortion so signal peaks stay
    exactly where they were in time. Critical for EEG, ECG,
    and any analysis where timing matters.

    Design: every method returns self so chaining works:
        Data(ecg, fs=1000).notch(50).bandpass(0.5, 40).hilbert()
    """

    def _require_fs(self, method_name: str) -> None:
        if self.fs is None:
            raise ValueError(
                f"{method_name} requires fs. "
                f"Set it when creating Data: Data(signal, fs=1000)"
            )

    # ── 1. Butterworth lowpass ─────────────────────────────────────────────

    def lowpass(
        self,
        cutoff: float,
        order: int = 4,
        zero_phase: bool = True,
    ) -> "FilterMixin":
        """
        Remove frequencies ABOVE cutoff.

        Use cases:
          - Remove EMG noise from ECG (cutoff ~40 Hz)
          - Smooth a signal while preserving slow trends
          - Anti-aliasing before downsampling

        Parameters
        ----------
        cutoff     : cutoff frequency in Hz
        order      : filter steepness (4 is standard; higher = sharper but
                     may ring on transients — don't go above 8)
        zero_phase : True = filtfilt (no phase shift, recommended)
                     False = causal lfilter (real-time processing only)

        Example
        -------
        Data(eeg, fs=256).lowpass(cutoff=40)
        """
        self._require_fs("lowpass")
        nyq = self.fs / 2.0
        if cutoff >= nyq:
            raise ValueError(
                f"Cutoff {cutoff} Hz must be below Nyquist ({nyq} Hz)"
            )

        sos = butter(order, cutoff / nyq, btype="low", output="sos")
        self.data = (
            sosfiltfilt(sos, self.data)
            if zero_phase
            else self._sosfilt_causal(sos, self.data)
        )

        self.filter_result = FilterResult(
            filter_type="Butterworth lowpass",
            cutoff=cutoff, order=order,
            fs=self.fs, zero_phase=zero_phase, label=self.label,
        )
        self._record(f"lowpass(cutoff={cutoff}Hz, order={order})")
        return self

    # ── 2. Butterworth highpass ────────────────────────────────────────────

    def highpass(
        self,
        cutoff: float,
        order: int = 4,
        zero_phase: bool = True,
    ) -> "FilterMixin":
        """
        Remove frequencies BELOW cutoff.

        Use cases:
          - Remove slow baseline drift from ECG (cutoff ~0.5 Hz)
          - Remove DC offset and slow wandering
          - Isolate action potentials in neural spike data

        Parameters
        ----------
        cutoff : cutoff frequency in Hz (typically 0.5–5 Hz for biomedical)

        Example
        -------
        Data(ecg, fs=1000).highpass(cutoff=0.5)   # remove baseline wander
        """
        self._require_fs("highpass")
        nyq = self.fs / 2.0
        if cutoff >= nyq:
            raise ValueError(f"Cutoff {cutoff} Hz must be below Nyquist ({nyq} Hz)")

        sos = butter(order, cutoff / nyq, btype="high", output="sos")
        self.data = (
            sosfiltfilt(sos, self.data)
            if zero_phase
            else self._sosfilt_causal(sos, self.data)
        )

        self.filter_result = FilterResult(
            filter_type="Butterworth highpass",
            cutoff=cutoff, order=order,
            fs=self.fs, zero_phase=zero_phase, label=self.label,
        )
        self._record(f"highpass(cutoff={cutoff}Hz, order={order})")
        return self

    # ── 3. Butterworth bandpass ────────────────────────────────────────────

    def bandpass(
        self,
        low: float,
        high: float,
        order: int = 4,
        zero_phase: bool = True,
    ) -> "FilterMixin":
        """
        Keep only frequencies BETWEEN low and high. Everything outside
        is attenuated.

        EEG frequency bands (standard):
          Delta  : 0.5 –  4 Hz  (deep sleep)
          Theta  : 4   –  8 Hz  (drowsiness, meditation)
          Alpha  : 8   – 13 Hz  (relaxed wakefulness)
          Beta   : 13  – 30 Hz  (active thinking)
          Gamma  : 30  – 80 Hz  (high cognition)

        ECG: bandpass(0.5, 40) removes both drift and EMG noise.
        EMG: bandpass(20, 450) isolates muscle activation.
        FTIR: used for isolating specific absorption band regions.

        Parameters
        ----------
        low, high : frequency band edges in Hz

        Example
        -------
        Data(eeg, fs=256).bandpass(8, 13)   # isolate alpha band
        """
        self._require_fs("bandpass")
        nyq = self.fs / 2.0
        if low <= 0 or high >= nyq:
            raise ValueError(
                f"Band [{low}, {high}] Hz must be within (0, {nyq}) Hz"
            )
        if low >= high:
            raise ValueError(f"low ({low}) must be less than high ({high})")

        sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
        self.data = (
            sosfiltfilt(sos, self.data)
            if zero_phase
            else self._sosfilt_causal(sos, self.data)
        )

        self.filter_result = FilterResult(
            filter_type="Butterworth bandpass",
            cutoff=(low, high), order=order,
            fs=self.fs, zero_phase=zero_phase, label=self.label,
        )
        self._record(f"bandpass(low={low}Hz, high={high}Hz, order={order})")
        return self

    # ── 4. Bandstop (notch) ────────────────────────────────────────────────

    def bandstop(
        self,
        low: float,
        high: float,
        order: int = 4,
        zero_phase: bool = True,
    ) -> "FilterMixin":
        """
        Remove a specific frequency BAND (wider notch than notch()).

        Use when you need to remove a wide noise band rather than
        a sharp single frequency. For surgical single-frequency
        removal, use notch() instead.

        Parameters
        ----------
        low, high : edges of the band to remove in Hz

        Example
        -------
        Data(signal, fs=1000).bandstop(45, 55)  # wide power-line removal
        """
        self._require_fs("bandstop")
        nyq = self.fs / 2.0

        sos = butter(order, [low / nyq, high / nyq], btype="bandstop", output="sos")
        self.data = (
            sosfiltfilt(sos, self.data)
            if zero_phase
            else self._sosfilt_causal(sos, self.data)
        )

        self.filter_result = FilterResult(
            filter_type="Butterworth bandstop",
            cutoff=(low, high), order=order,
            fs=self.fs, zero_phase=zero_phase, label=self.label,
        )
        self._record(f"bandstop(low={low}Hz, high={high}Hz)")
        return self

    # ── 5. Notch filter ────────────────────────────────────────────────────

    def notch(
        self,
        freq: float,
        quality: float = 30.0,
        harmonics: bool = False,
        zero_phase: bool = True,
    ) -> "FilterMixin":
        """
        Surgically remove a single frequency (like a laser cut).

        This is the standard tool for removing power-line interference:
          - 50 Hz in Europe, Asia, Africa, India
          - 60 Hz in North America, parts of South America

        How it works:
          iirnotch designs an IIR notch filter with a very narrow
          rejection band centred on freq. The quality factor (Q)
          controls the width: higher Q = narrower notch = more
          surgical but more sensitive to exact frequency.

        Parameters
        ----------
        freq      : frequency to remove in Hz (50 or 60 typically)
        quality   : Q factor. Higher = narrower notch.
                    Q=30 is the standard clinical default.
                    Q=10 for slightly drifting interference.
        harmonics : if True, also notch 2×freq, 3×freq, 4×freq.
                    Power lines produce harmonics (100Hz, 150Hz etc.)
                    that can also contaminate the signal.
        zero_phase : True = filtfilt (recommended)

        Example
        -------
        Data(ecg, fs=1000).notch(50)                    # India/EU
        Data(ecg, fs=1000).notch(60, harmonics=True)    # USA + overtones
        Data(eeg, fs=256).notch(50).bandpass(1, 100)    # standard EEG chain
        """
        self._require_fs("notch")
        nyq = self.fs / 2.0

        freqs_to_notch = [freq]
        if harmonics:
            k = 2
            while freq * k < nyq:
                freqs_to_notch.append(freq * k)
                k += 1

        for f in freqs_to_notch:
            w0 = f / nyq
            b, a = iirnotch(w0, quality)
            if zero_phase:
                self.data = filtfilt(b, a, self.data)
            else:
                from scipy.signal import lfilter
                self.data = lfilter(b, a, self.data)

        self.filter_result = FilterResult(
            filter_type=f"Notch (IIR)",
            cutoff=freqs_to_notch, order=2,
            fs=self.fs, zero_phase=zero_phase, label=self.label,
        )
        notch_str = f"notch(freq={freq}Hz, Q={quality}, harmonics={harmonics})"
        self._record(notch_str)
        return self

    # ── 6. FIR filter (window method) ─────────────────────────────────────

    def fir_filter(
        self,
        cutoff,
        filter_type: str = "lowpass",
        numtaps: int = 101,
        window: str = "hamming",
    ) -> "FilterMixin":
        """
        FIR (Finite Impulse Response) filter — linear phase guaranteed.

        When to prefer FIR over Butterworth (IIR):
          - When you need guaranteed linear phase (group delay constant
            across all frequencies — important for preserving waveform
            shape in ERPs and spike sorting).
          - When filter stability is critical (FIR is always stable).

        When to prefer Butterworth:
          - When you need a steeper rolloff for the same number of taps.
          - When computational cost matters.

        Parameters
        ----------
        cutoff      : float or [low, high] in Hz
        filter_type : 'lowpass', 'highpass', 'bandpass', 'bandstop'
        numtaps     : filter length. Odd number required. More taps =
                      sharper cutoff but more latency.
        window      : window function for FIR design.
                      'hamming'   — good sidelobe suppression (default)
                      'blackman'  — better stopband, wider transition
                      'hann'      — smooth spectral roll-off

        Example
        -------
        Data(erp, fs=1000).fir_filter([1, 40], filter_type="bandpass")
        """
        self._require_fs("fir_filter")
        nyq = self.fs / 2.0

        if isinstance(cutoff, (int, float)):
            cutoff_norm = cutoff / nyq
        else:
            cutoff_norm = [c / nyq for c in cutoff]

        if numtaps % 2 == 0:
            numtaps += 1     # FIR requires odd tap count for Type I

        taps = firwin(numtaps, cutoff_norm, window=window, pass_zero=(
            filter_type in ("lowpass", "bandstop")
        ))

        self.data = filtfilt(taps, [1.0], self.data)

        self.filter_result = FilterResult(
            filter_type=f"FIR {filter_type}",
            cutoff=cutoff, order=numtaps,
            fs=self.fs, zero_phase=True, label=self.label,
        )
        self._record(f"fir_filter(cutoff={cutoff}, type={filter_type}, taps={numtaps})")
        return self

    # ── 7. Savitzky-Golay smoothing ───────────────────────────────────────

    def savgol(
        self,
        window_length: int = 11,
        polyorder: int = 3,
    ) -> "FilterMixin":
        """
        Savitzky-Golay smoothing filter.

        Unlike Butterworth, SavGol fits a local polynomial to each
        window rather than attenuating frequencies. This preserves
        peak shapes and amplitudes much better than lowpass filtering.

        Ideal for:
          - FTIR and Raman spectral smoothing (standard in chemistry)
          - Smoothing ECG peaks without distorting the QRS complex
          - Any signal where preserving peak position + height matters

        Parameters
        ----------
        window_length : number of samples in the sliding window.
                        Must be odd and > polyorder.
                        Larger = smoother but risks oversmoothing peaks.
        polyorder     : degree of the fitting polynomial.
                        3 or 4 is standard for biomedical signals.

        Example
        -------
        Data(ftir, fs=1).savgol(window_length=15, polyorder=3)
        """
        from scipy.signal import savgol_filter
        if window_length % 2 == 0:
            window_length += 1
        if window_length <= polyorder:
            raise ValueError("window_length must be greater than polyorder")

        self.data = savgol_filter(self.data, window_length, polyorder)
        self._record(f"savgol(window={window_length}, poly={polyorder})")
        return self

    # ── 8. Resampling ─────────────────────────────────────────────────────

    def resample(self, target_fs: float) -> "FilterMixin":
        """
        Resample the signal to a new sampling frequency.

        Uses polyphase filtering (resample_poly) — the gold standard.
        Automatically applies anti-aliasing when downsampling so you
        don't get aliasing artifacts.

        Use cases:
          - Downsampling EEG from 2048 Hz to 256 Hz (saves 8× memory
            and compute for downstream ML)
          - Aligning two signals recorded at different rates
          - Standardising dataset sampling rates before batch processing

        Parameters
        ----------
        target_fs : desired sampling frequency in Hz

        Example
        -------
        Data(eeg, fs=2048).resample(256)   # 8x downsample
        """
        self._require_fs("resample")
        from math import gcd

        orig_fs = int(round(self.fs))
        tgt_fs  = int(round(target_fs))
        g       = gcd(orig_fs, tgt_fs)
        up, down = tgt_fs // g, orig_fs // g

        self.data = resample_poly(self.data, up, down).astype(float)

        old_fs   = self.fs
        self.fs  = target_fs
        self._record(f"resample({old_fs}Hz → {target_fs}Hz)")
        return self

    # ── 9. Filter visualiser ──────────────────────────────────────────────

    def plot_filter_response(
        self,
        cutoff,
        filter_type: str = "bandpass",
        order: int = 4,
        save_path: str = None,
    ) -> "FilterMixin":
        """
        Plot the frequency response of a filter BEFORE applying it.
        Lets you verify the filter is doing what you expect.

        Parameters
        ----------
        cutoff      : same as the filter methods
        filter_type : 'lowpass', 'highpass', 'bandpass', 'notch'
        order       : Butterworth filter order
        """
        self._require_fs("plot_filter_response")
        import matplotlib.pyplot as plt
        from scipy.signal import freqz

        nyq = self.fs / 2.0

        if filter_type == "notch":
            from scipy.signal import iirnotch
            w0 = cutoff / nyq
            b, a = iirnotch(w0, 30)
        else:
            btype_map = {
                "lowpass":  "low",
                "highpass": "high",
                "bandpass": "band",
                "bandstop": "bandstop",
            }
            btype = btype_map.get(filter_type, "band")
            if isinstance(cutoff, (int, float)):
                wn = cutoff / nyq
            else:
                wn = [c / nyq for c in cutoff]
            b, a = butter(order, wn, btype=btype)

        w, h = freqz(b, a, worN=4096, fs=self.fs)
        magnitude_db = 20 * np.log10(np.abs(h) + 1e-12)

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        fig.patch.set_facecolor("white")
        fig.suptitle(f"Filter response — {filter_type}  cutoff={cutoff} Hz",
                     fontsize=12, fontweight="bold")

        # Magnitude
        ax = axes[0]
        ax.plot(w, magnitude_db, color="#5B4FCF", lw=1.5)
        ax.axhline(-3, color="#D85A30", ls="--", lw=1, label="-3 dB cutoff")
        ax.axhline(-40, color="#888780", ls=":", lw=0.8, label="-40 dB")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title("Magnitude response")
        ax.legend(fontsize=9, frameon=False)
        ax.set_facecolor("#f8f8f8")
        ax.grid(True, color="white")
        for sp in ax.spines.values(): sp.set_visible(False)

        # Phase
        ax2 = axes[1]
        ax2.plot(w, np.angle(h, deg=True), color="#1D9E75", lw=1.5)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (degrees)")
        ax2.set_title("Phase response")
        ax2.set_facecolor("#f8f8f8")
        ax2.grid(True, color="white")
        for sp in ax2.spines.values(): sp.set_visible(False)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
        else:
            plt.show()
        return self

    # ── internal helper ───────────────────────────────────────────────────

    def _sosfilt_causal(self, sos, x):
        from scipy.signal import sosfilt
        return sosfilt(sos, x)