from __future__ import annotations
import numpy as np
from PBstats.core.base import WaveletResult

class WaveletMixin:

    def wavelet(
        self,
        wavelet: str = "cmor1.5-1.0",
        n_scales: int = 64,
        freq_min: float = 1.0,
        freq_max: float = None,
    ) -> "WaveletMixin":
        """
        Continuous Wavelet Transform (CWT) for time-frequency analysis.

        Better than FFT when your signal is non-stationary
        (e.g. EEG bursts, transient FTIR features, heart rate variability).

        Parameters
        ----------
        wavelet   : PyWavelets CWT wavelet. Good choices:
                    'cmor1.5-1.0'  — complex Morlet (default, most used in biomedical)
                    'gaus1'        — Gaussian derivative
                    'mexh'         — Mexican hat (good for peak detection)
        n_scales  : number of frequency bins (resolution)
        freq_min  : lowest frequency of interest in Hz
        freq_max  : highest frequency (default: fs/2)
        """
        try:
            import pywt
        except ImportError:
            raise ImportError("PyWavelets required: pip install PyWavelets")

        if self.fs is None:
            raise ValueError("fs is required for wavelet transform.")

        freq_max = freq_max or (self.fs / 2)

        # Convert frequency range to scales
        # scale = fs × central_freq_of_wavelet / frequency
        central_freq = pywt.central_frequency(wavelet)
        freqs = np.linspace(freq_min, freq_max, n_scales)
        scales = (central_freq * self.fs) / freqs

        coefficients, _ = pywt.cwt(
            self.data, scales, wavelet, sampling_period=1.0 / self.fs
        )

        times = np.arange(len(self.data)) / self.fs

        self.wavelet_result = WaveletResult(
            coefficients=coefficients,
            freqs=freqs,
            times=times,
            scales=scales,
            wavelet=wavelet,
            label=self.label,
        )

        # Update self.data to mean power across scales
        self.data = np.mean(np.abs(coefficients), axis=0)
        self._record(f"wavelet(wavelet={wavelet}, n_scales={n_scales})")
        return self