from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class FFTResult:
    freqs: np.ndarray        # frequency axis in Hz
    magnitude: np.ndarray   # magnitude spectrum
    phase: np.ndarray        # phase spectrum in radians
    power: np.ndarray        # power spectrum (magnitude²)
    fs: float                # sampling frequency used
    label: str = ""

    def peak_frequency(self) -> float:
        """Return the frequency with highest magnitude."""
        return float(self.freqs[np.argmax(self.magnitude)])

    def band_power(self, low: float, high: float) -> float:
        """Total power within a frequency band [low, high] Hz."""
        mask = (self.freqs >= low) & (self.freqs <= high)
        return float(self.power[mask].sum())

    def dominant_bands(self, n: int = 3) -> list[tuple[float, float]]:
        """Return the n frequencies with highest power."""
        idx = np.argsort(self.magnitude)[::-1][:n]
        return [(float(self.freqs[i]), float(self.magnitude[i])) for i in idx]


@dataclass
class HilbertResult:
    analytic: np.ndarray     # complex analytic signal
    envelope: np.ndarray     # amplitude envelope (instantaneous amplitude)
    phase: np.ndarray        # instantaneous phase (radians)
    frequency: np.ndarray    # instantaneous frequency (Hz)
    fs: float
    label: str = ""

    def mean_envelope(self) -> float:
        return float(self.envelope.mean())

    def peak_envelope(self) -> float:
        return float(self.envelope.max())


@dataclass
class WaveletResult:
    coefficients: np.ndarray  # 2D array: (scales, time)
    freqs: np.ndarray          # frequency axis
    times: np.ndarray          # time axis
    scales: np.ndarray
    wavelet: str
    label: str = ""

    def scalogram(self) -> np.ndarray:
        """Return power scalogram (|coefficients|²)."""
        return np.abs(self.coefficients) ** 2


@dataclass
class STFTResult:
    freqs: np.ndarray          # frequency axis
    times: np.ndarray          # time axis
    Zxx: np.ndarray            # complex STFT matrix
    magnitude: np.ndarray      # |Zxx|
    fs: float
    label: str = ""

    def spectrogram(self) -> np.ndarray:
        """Power spectrogram in dB."""
        power = np.abs(self.Zxx) ** 2
        return 10 * np.log10(power + 1e-12)