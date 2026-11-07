from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from PBstats.core.base import HilbertResult
from PBstats.core.steps import step
from PBstats.core.utils import check_length, require_1d


class HilbertMixin:

    @step
    def hilbert(self) -> "HilbertMixin":
        """
        Analytic signal via the Hilbert transform.

        Produces the amplitude envelope (bursts in EEG/EMG), instantaneous
        phase (synchrony analysis) and instantaneous frequency (Hz).
        ``self.data`` becomes the envelope; the full result, including the
        exact input that was transformed, is in ``self.hilbert_result``.

        Tip: band-pass first — the instantaneous phase/frequency are only
        meaningful for narrow-band signals.
        """
        self._require_fs("hilbert")
        x = require_1d(self.data, "hilbert")
        check_length(x, 2, "hilbert")

        analytic = hilbert(x)
        envelope = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.gradient(phase) / (2.0 * np.pi) * self.fs

        self.hilbert_result = HilbertResult(
            analytic=analytic,
            envelope=envelope,
            phase=phase,
            frequency=inst_freq,
            fs=self.fs,
            label=self.label,
            signal=x.copy(),
        )
        self.data = envelope
        self._record("hilbert()")
        return self
