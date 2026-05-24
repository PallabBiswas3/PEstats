from __future__ import annotations
import numpy as np
from scipy.signal import hilbert
from PBstats.core.base import HilbertResult

class HilbertMixin:

    def hilbert(self) -> "HilbertMixin":
        """
        Compute the analytic signal via Hilbert transform.

        Produces:
          - amplitude envelope  (useful for detecting signal bursts in EEG/EMG)
          - instantaneous phase (useful for phase synchrony analysis)
          - instantaneous frequency (rate of phase change, in Hz)

        self.data is updated to the envelope so chaining continues.
        Full result stored in self.hilbert_result.
        """
        if self.fs is None:
            raise ValueError("fs is required for instantaneous frequency calculation.")

        x = self.data
        analytic = hilbert(x)

        envelope    = np.abs(analytic)
        phase       = np.unwrap(np.angle(analytic))

        # Instantaneous frequency = derivative of phase / (2π) × fs
        inst_freq = np.diff(phase) / (2.0 * np.pi) * self.fs
        inst_freq = np.append(inst_freq, inst_freq[-1])   # match length

        self.hilbert_result = HilbertResult(
            analytic=analytic,
            envelope=envelope,
            phase=phase,
            frequency=inst_freq,
            fs=self.fs,
            label=self.label,
        )

        self.data = envelope
        self._record("hilbert()")
        return self