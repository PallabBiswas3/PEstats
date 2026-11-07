from __future__ import annotations

import numpy as np

from PBstats.core.steps import step


class ImputationMixin:
    """
    impute — missing-value handling with tabular-style methods.

    For signals, ``remove_missing()`` (interpolation) is usually best.
    ``impute()`` adds ``knn``, which fills a channel's gaps using the other
    channels (rows with similar values elsewhere) — useful for multichannel
    sensor arrays with dropouts on individual channels.
    """

    @step
    def impute(self, method: str = "mean", n_neighbors: int = 5) -> "ImputationMixin":
        """
        Fill NaNs column-by-column.

        Parameters
        ----------
        method : 'mean' | 'median' | 'ffill' | 'bfill' | 'interpolate' | 'zero'
                 | 'knn' (2D only; uses sklearn's KNNImputer)
        """
        if method == "knn":
            if self.data.ndim != 2:
                raise ValueError("impute(method='knn') needs multichannel (2D) data.")
            from sklearn.impute import KNNImputer
            n_missing = int(np.isnan(self.data).sum())
            self.data = KNNImputer(n_neighbors=n_neighbors).fit_transform(self.data)
            self._record(f"impute(method=knn, n_missing={n_missing})")
            return self

        # Every other method shares the per-channel implementation.
        self.remove_missing(strategy=method)
        return self
