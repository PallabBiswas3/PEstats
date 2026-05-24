from __future__ import annotations
import numpy as np

class CleaningMixin:
    """
    Methods: remove_missing, remove_dc, clip
    Each mutates self.data, records the step, returns self.
    """

    def remove_missing(self, strategy: str = "interpolate") -> "CleaningMixin":
        """
        Replace NaN values in the signal.

        Parameters
        ----------
        strategy : 'interpolate' — linear interpolation between neighbours
                   'zero'        — replace with 0
                   'mean'        — replace with signal mean
                   'drop'        — remove NaN positions entirely
        """
        x = self.data.astype(float)

        if strategy == "interpolate":
            nan_mask = np.isnan(x)
            if nan_mask.any():
                indices = np.arange(len(x))
                x[nan_mask] = np.interp(
                    indices[nan_mask],
                    indices[~nan_mask],
                    x[~nan_mask]
                )

        elif strategy == "zero":
            x = np.nan_to_num(x, nan=0.0)

        elif strategy == "mean":
            mean_val = np.nanmean(x)
            x = np.where(np.isnan(x), mean_val, x)

        elif strategy == "drop":
            x = x[~np.isnan(x)]

        else:
            raise ValueError(f"Unknown strategy '{strategy}'. "
                             f"Choose: interpolate, zero, mean, drop")

        self.data = x
        self._record(f"remove_missing(strategy={strategy})")
        return self

    def remove_dc(self) -> "CleaningMixin":
        """
        Subtract the mean (remove DC offset).
        Essential for spectral analysis — DC shows up as a massive
        spike at 0 Hz in FFT if not removed.
        """
        self.data = self.data - np.mean(self.data)
        self._record("remove_dc()")
        return self

    def clip(self, low: float, high: float) -> "CleaningMixin":
        """
        Hard-clip values to [low, high].
        Useful for removing sensor saturation artifacts.
        """
        self.data = np.clip(self.data, low, high)
        self._record(f"clip(low={low}, high={high})")
        return self