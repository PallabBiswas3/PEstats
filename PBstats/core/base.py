"""Result containers for transforms. Single source of truth — re-exported by
``PBstats.transforms``."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import trapezoid


@dataclass
class FFTResult:
    freqs: np.ndarray        # frequency axis in Hz
    magnitude: np.ndarray    # amplitude spectrum (signal units if normalised)
    phase: np.ndarray        # phase spectrum in radians
    power: np.ndarray        # magnitude²
    fs: float                # sampling frequency used
    label: str = ""

    def peak_frequency(self, exclude_dc: bool = True) -> float:
        """Frequency with the highest magnitude (DC bin ignored by default)."""
        mag = self.magnitude.copy()
        if exclude_dc and len(mag) > 1:
            mag[0] = -np.inf
        return float(self.freqs[np.argmax(mag)])

    def band_power(self, low: float, high: float) -> float:
        """Summed power of bins within [low, high] Hz."""
        mask = (self.freqs >= low) & (self.freqs <= high)
        return float(self.power[mask].sum())

    def dominant_bands(self, n: int = 3) -> list[tuple[float, float]]:
        """The n (frequency, magnitude) pairs with the highest magnitude."""
        idx = np.argsort(self.magnitude)[::-1][:n]
        return [(float(self.freqs[i]), float(self.magnitude[i])) for i in idx]


@dataclass
class PSDResult:
    """Welch power spectral density — a low-variance spectrum estimate."""
    freqs: np.ndarray        # Hz
    psd: np.ndarray          # power / Hz
    fs: float
    nperseg: int
    label: str = ""

    @property
    def resolution(self) -> float:
        """Frequency resolution in Hz."""
        return float(self.fs / self.nperseg)

    def band_power(self, low: float, high: float) -> float:
        """Absolute power in [low, high] Hz (integrated, signal units²)."""
        mask = (self.freqs >= low) & (self.freqs <= high)
        if mask.sum() < 2:
            return float(self.psd[mask].sum() * self.resolution)
        return float(trapezoid(self.psd[mask], self.freqs[mask]))

    def total_power(self) -> float:
        return float(trapezoid(self.psd, self.freqs))

    def relative_band_power(self, low: float, high: float) -> float:
        """Fraction of total power in [low, high] Hz."""
        return self.band_power(low, high) / (self.total_power() + 1e-300)

    def peak_frequency(self, fmin: float = 0.0, fmax: float | None = None) -> float:
        mask = self.freqs >= fmin
        if fmax is not None:
            mask &= self.freqs <= fmax
        if not mask.any():
            raise ValueError(f"No frequency bins in [{fmin}, {fmax}] Hz")
        f, p = self.freqs[mask], self.psd[mask]
        return float(f[np.argmax(p)])

    def to_db(self) -> np.ndarray:
        return 10 * np.log10(self.psd + 1e-300)


@dataclass
class HilbertResult:
    analytic: np.ndarray     # complex analytic signal
    envelope: np.ndarray     # instantaneous amplitude
    phase: np.ndarray        # unwrapped instantaneous phase (rad)
    frequency: np.ndarray    # instantaneous frequency (Hz)
    fs: float
    label: str = ""
    signal: np.ndarray = field(default=None, repr=False)  # the input transformed

    def mean_envelope(self) -> float:
        return float(self.envelope.mean())

    def peak_envelope(self) -> float:
        return float(self.envelope.max())


@dataclass
class WaveletResult:
    coefficients: np.ndarray  # (scales, time)
    freqs: np.ndarray
    times: np.ndarray
    scales: np.ndarray
    wavelet: str
    label: str = ""

    def scalogram(self) -> np.ndarray:
        """Power scalogram |coefficients|²."""
        return np.abs(self.coefficients) ** 2


@dataclass
class STFTResult:
    freqs: np.ndarray
    times: np.ndarray
    Zxx: np.ndarray            # complex STFT matrix
    magnitude: np.ndarray      # |Zxx|
    fs: float
    label: str = ""

    def spectrogram(self) -> np.ndarray:
        """Power spectrogram in dB."""
        return 10 * np.log10(np.abs(self.Zxx) ** 2 + 1e-12)
