from __future__ import annotations
import numpy as np
from scipy.signal import stft
from PBstats.core.base import STFTResult

class STFTMixin:

    def stft(
        self,
        window: str = "hann",
        nperseg: int = 256,
        noverlap: int = None,
    ) -> "STFTMixin":
        """
        Short-Time Fourier Transform — sliding window FFT.

        Use this over plain FFT when you need to see how
        the frequency content changes over time (non-stationary signals).
        The result is a 2D time-frequency matrix.

        Parameters
        ----------
        window   : window function (same options as fft())
        nperseg  : samples per segment (controls time/freq resolution tradeoff)
                   larger → better frequency resolution, worse time resolution
                   smaller → better time resolution, worse frequency resolution
        noverlap : overlap between segments (default: nperseg // 2)
        """
        if self.fs is None:
            raise ValueError("fs is required for STFT.")

        noverlap = noverlap if noverlap is not None else nperseg // 2

        freqs, times, Zxx = stft(
            self.data,
            fs=self.fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
        )

        self.stft_result = STFTResult(
            freqs=freqs,
            times=times,
            Zxx=Zxx,
            magnitude=np.abs(Zxx),
            fs=self.fs,
            label=self.label,
        )

        self.data = np.abs(Zxx).mean(axis=1)   # mean power per frequency bin
        self._record(f"stft(nperseg={nperseg}, noverlap={noverlap})")
        return self