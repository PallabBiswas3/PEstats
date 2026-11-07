from __future__ import annotations

import numpy as np
from scipy.signal import stft

from PBstats.core.base import STFTResult
from PBstats.core.steps import step
from PBstats.core.utils import require_1d


class STFTMixin:

    @step
    def stft(
        self,
        window: str = "hann",
        nperseg: int = 256,
        noverlap: int | None = None,
    ) -> "STFTMixin":
        """
        Short-Time Fourier Transform — sliding-window FFT for non-stationary
        signals. ``self.data`` becomes the mean magnitude per frequency bin.

        Parameters
        ----------
        nperseg  : samples per segment (larger → finer frequency resolution,
                   coarser time resolution). Capped at the signal length.
        noverlap : overlap between segments (default nperseg // 2)
        """
        self._require_fs("stft")
        x = require_1d(self.data, "stft")
        nperseg = int(min(nperseg, len(x)))
        noverlap = noverlap if noverlap is not None else nperseg // 2
        if noverlap >= nperseg:
            raise ValueError(f"noverlap ({noverlap}) must be < nperseg ({nperseg})")

        freqs, times, Zxx = stft(
            x, fs=self.fs, window=window, nperseg=nperseg, noverlap=noverlap,
        )
        self.stft_result = STFTResult(
            freqs=freqs, times=times, Zxx=Zxx,
            magnitude=np.abs(Zxx), fs=self.fs, label=self.label,
        )
        self.data = np.abs(Zxx).mean(axis=1)
        self._record(f"stft(nperseg={nperseg}, noverlap={noverlap})")
        return self
