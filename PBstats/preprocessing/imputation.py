from __future__ import annotations
import numpy as np

class ImputationMixin:
    """
    Methods: impute
    More sophisticated missing value handling for 2D (tabular) data.
    For 1D signals, remove_missing() in CleaningMixin is sufficient.
    """

    def impute(self, method: str = "mean") -> "ImputationMixin":
        """
        Fill NaNs in a 2D array column-by-column.

        Parameters
        ----------
        method : 'mean'   — fill with column mean
                 'median' — fill with column median
                 'ffill'  — forward-fill (carry last valid value forward)
                 'bfill'  — backward-fill
        """
        if self.data.ndim == 1:
            # Redirect 1D to the simpler cleaner
            return self.remove_missing(
                strategy="interpolate" if method in ("ffill","bfill") else method
            )

        x = self.data.astype(float)

        for col in range(x.shape[1]):
            column = x[:, col]
            nan_mask = np.isnan(column)
            if not nan_mask.any():
                continue

            if method == "mean":
                column[nan_mask] = np.nanmean(column)
            elif method == "median":
                column[nan_mask] = np.nanmedian(column)
            elif method == "ffill":
                # Vectorized ffill: mask valid indices, then use ffill logic
                mask = ~nan_mask
                idx = np.where(mask, np.arange(len(column)), 0)
                np.maximum.accumulate(idx, out=idx)
                column[:] = column[idx]
            elif method == "bfill":
                # Vectorized bfill: reverse, ffill, reverse
                mask = ~nan_mask
                idx = np.where(mask, np.arange(len(column)), len(column) - 1)
                # Reverse accumulation for bfill
                idx = len(column) - 1 - np.maximum.accumulate((len(column) - 1 - idx)[::-1])[::-1]
                column[:] = column[idx]
            else:
                raise ValueError(f"Unknown method '{method}'. "
                                 f"Choose: mean, median, ffill, bfill")
            x[:, col] = column

        self.data = x
        self._record(f"impute(method={method})")
        return self