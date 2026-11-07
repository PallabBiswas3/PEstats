from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window, welch

from PBstats.core.base import FFTResult, PSDResult
from PBstats.core.steps import step
from PBstats.core.utils import require_1d


class FFTMixin:

    @step
    def fft(self, window: str = "hann", normalise: bool = True) -> "FFTMixin":
        """
        Real FFT of the signal. Result in ``self.fft_result``.

        ``self.data`` becomes the magnitude spectrum so a chain can continue
        in the frequency domain.

        Parameters
        ----------
        window    : scipy window applied before the FFT to reduce leakage
                    ('hann', 'hamming', 'blackman', 'boxcar' = none, ...)
        normalise : if True, magnitudes are single-sided *amplitudes* in signal
                    units — a sine of amplitude A shows a peak of ~A regardless
                    of window or length (window coherent gain is corrected).
        """
        self._require_fs("fft")
        x = require_1d(self.data, "fft")
        n = len(x)

        win = self._make_window(window, n)
        spectrum = rfft(x * win)
        freqs = rfftfreq(n, d=1.0 / self.fs)

        magnitude = np.abs(spectrum)
        if normalise:
            magnitude = magnitude / win.sum()
            # single-sided: double every bin except DC (and Nyquist for even n)
            magnitude[1:] *= 2.0
            if n % 2 == 0:
                magnitude[-1] /= 2.0

        self.fft_result = FFTResult(
            freqs=freqs,
            magnitude=magnitude,
            phase=np.angle(spectrum),
            power=magnitude ** 2,
            fs=self.fs,
            label=self.label,
        )
        self.data = magnitude
        self._record(f"fft(window={window}, normalise={normalise})")
        return self

    @step
    def psd(
        self,
        nperseg: int | None = None,
        window: str = "hann",
        overlap: float = 0.5,
    ) -> "FFTMixin":
        """
        Welch power spectral density — averages periodograms of overlapping
        segments, giving a far less noisy spectrum than a single FFT.
        This is the right tool for band power, peak frequency and spectral
        features. Non-destructive: ``self.data`` is unchanged.

        Parameters
        ----------
        nperseg : samples per segment. Default: 2 s of data (0.5 Hz
                  resolution), capped at the signal length.
        overlap : fractional segment overlap (0–1).
        """
        self._require_fs("psd")
        x = require_1d(self.data, "psd")
        nperseg = int(min(nperseg or max(int(2 * self.fs), 8), len(x)))
        freqs, pxx = welch(
            x, fs=self.fs, window=window, nperseg=nperseg,
            noverlap=int(nperseg * overlap),
        )
        self.psd_result = PSDResult(
            freqs=freqs, psd=pxx, fs=self.fs, nperseg=nperseg, label=self.label,
        )
        self._record(f"psd(nperseg={nperseg}, window={window})")
        return self

    @staticmethod
    def _make_window(name: str, n: int) -> np.ndarray:
        try:
            return get_window(name, n)
        except ValueError:
            raise ValueError(
                f"Unknown window '{name}'. Try: hann, hamming, blackman, bartlett, boxcar"
            ) from None
