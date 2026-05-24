from __future__ import annotations
import numpy as np
from scipy.fft import rfft, rfftfreq
from PBstats.core.base import FFTResult 

class FFTMixin:

    def fft(
        self,
        window: str = "hann",
        normalise: bool = True,
    ) -> "FFTMixin":
        """
        Compute the real FFT of the signal and store result on self.fft_result.
        Also updates self.data to the magnitude spectrum so chaining continues.

        Parameters
        ----------
        window    : scipy window name applied before FFT to reduce spectral leakage.
                    Common choices: 'hann', 'hamming', 'blackman', 'boxcar' (none)
        normalise : if True, divide magnitude by N so amplitude is in signal units
        """
        if self.fs is None:
            raise ValueError(
                "fs (sampling frequency) is required for FFT. "
                "Set it when creating Data: Data(signal, fs=1000)"
            )

        x = self.data.copy()
        n = len(x)

        # Apply window
        win = self._make_window(window, n)
        x_windowed = x * win

        # Compute FFT
        spectrum = rfft(x_windowed)
        freqs = rfftfreq(n, d=1.0 / self.fs)

        magnitude = np.abs(spectrum)
        if normalise:
            magnitude = magnitude / (n / 2)   # convert to peak amplitude

        phase = np.angle(spectrum)
        power = magnitude ** 2

        self.fft_result = FFTResult(
            freqs=freqs,
            magnitude=magnitude,
            phase=phase,
            power=power,
            fs=self.fs,
            label=self.label,
        )

        # Update self.data so the chain can continue on the spectrum
        self.data = magnitude
        self._record(f"fft(window={window}, normalise={normalise})")
        return self

    def _make_window(self, name: str, n: int) -> np.ndarray:
        from scipy.signal import get_window
        try:
            return get_window(name, n)
        except ValueError:
            raise ValueError(
                f"Unknown window '{name}'. "
                f"Try: hann, hamming, blackman, bartlett, boxcar"
            )