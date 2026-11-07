from __future__ import annotations

import numpy as np

from PBstats.core.base import WaveletResult
from PBstats.core.steps import step
from PBstats.core.utils import require_1d


class WaveletMixin:

    @step
    def wavelet(
        self,
        wavelet: str = "cmor1.5-1.0",
        n_scales: int = 64,
        freq_min: float = 1.0,
        freq_max: float | None = None,
        spacing: str = "log",
    ) -> "WaveletMixin":
        """
        Continuous Wavelet Transform (CWT) for time-frequency analysis of
        non-stationary signals (EEG bursts, transients, HRV).
        ``self.data`` becomes the mean |coefficient| over scales.

        Parameters
        ----------
        wavelet  : PyWavelets CWT wavelet — 'cmor1.5-1.0' (complex Morlet,
                   default), 'gaus1', 'mexh', ...
        n_scales : number of frequency rows
        freq_min / freq_max : analysed band in Hz (default max: fs/2)
        spacing  : 'log' (default; constant relative resolution, the natural
                   choice for wavelets) or 'linear'
        """
        try:
            import pywt
        except ImportError:  # pragma: no cover
            raise ImportError("PyWavelets required: pip install PyWavelets") from None

        self._require_fs("wavelet")
        x = require_1d(self.data, "wavelet")

        freq_max = freq_max or (self.fs / 2)
        if not 0 < freq_min < freq_max:
            raise ValueError(f"Need 0 < freq_min < freq_max, got {freq_min}, {freq_max}")
        if spacing == "log":
            freqs = np.geomspace(freq_min, freq_max, n_scales)
        elif spacing == "linear":
            freqs = np.linspace(freq_min, freq_max, n_scales)
        else:
            raise ValueError("spacing must be 'log' or 'linear'")

        scales = pywt.frequency2scale(wavelet, freqs / self.fs)
        coefficients, _ = pywt.cwt(x, scales, wavelet, sampling_period=1.0 / self.fs)

        self.wavelet_result = WaveletResult(
            coefficients=coefficients,
            freqs=freqs,
            times=np.arange(len(x)) / self.fs,
            scales=scales,
            wavelet=wavelet,
            label=self.label,
        )
        self.data = np.mean(np.abs(coefficients), axis=0)
        self._record(f"wavelet(wavelet={wavelet}, n_scales={n_scales})")
        return self
