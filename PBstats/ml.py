"""
Machine-learning bridge: raw signals → feature matrix, the scikit-learn way.

    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from PBstats import Pipeline, SignalFeatureExtractor

    clean = Pipeline("clean").add("bandpass", low=0.5, high=40).add("standardize")
    model = make_pipeline(
        SignalFeatureExtractor(fs=250, preprocess=clean, include=["time", "freq"]),
        RandomForestClassifier(),
    )
    cross_val_score(model, X_signals, y, cv=5)

Because preprocessing + feature extraction live *inside* the estimator,
cross-validation stays honest and the whole model pickles as one object.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from PBstats.features import functions as F


def _as_signal_list(X) -> list[np.ndarray]:
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy(dtype=float)
    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            return [X.astype(float)]
        if X.ndim != 2:
            raise ValueError("X must be (n_signals, n_samples) or a list of 1D signals")
        return [row.astype(float) for row in X]
    return [np.asarray(s, dtype=float).flatten() for s in X]   # ragged list


def _extract_one(x, fs, include, preprocess) -> dict:
    if preprocess is not None:
        from PBstats.core.data import Data
        d = preprocess.run(Data(x, fs=fs))
        x, fs = d.data, d.fs          # a resample() step may change fs
    return F.extract(x, fs, include)


def extract_features(
    signals,
    fs: float,
    labels: Optional[Sequence] = None,
    include: Optional[Iterable[str]] = None,
    preprocess=None,
    n_jobs: Optional[int] = None,
    on_error: str = "raise",
) -> pd.DataFrame:
    """
    Build a feature table from many signals in one call.

    Parameters
    ----------
    signals    : 2D array (n_signals, n_samples) or list of 1D arrays
                 (lengths may differ)
    fs         : sampling frequency (Hz)
    labels     : optional target / id per signal → added as a 'label' column
    include    : feature categories ('time', 'freq', 'nonlinear', 'peaks', 'hrv')
    preprocess : optional PBstats ``Pipeline`` applied to each signal first
    n_jobs     : parallel workers (joblib); -1 = all cores
    on_error   : 'raise' or 'nan' (a failing signal becomes a row of NaN)

    Returns
    -------
    DataFrame, one row per signal.
    """
    from joblib import Parallel, delayed

    sigs = _as_signal_list(signals)
    if labels is not None and len(labels) != len(sigs):
        raise ValueError(f"{len(labels)} labels for {len(sigs)} signals")
    if on_error not in ("raise", "nan"):
        raise ValueError("on_error must be 'raise' or 'nan'")

    def _safe(x):
        try:
            return _extract_one(x, fs, include, preprocess)
        except Exception:
            if on_error == "raise":
                raise
            return {}

    rows = Parallel(n_jobs=n_jobs)(delayed(_safe)(x) for x in sigs)
    df = pd.DataFrame(rows)
    if labels is not None:
        df.insert(0, "label", list(labels))
    return df


class SignalFeatureExtractor(TransformerMixin, BaseEstimator):
    """
    scikit-learn transformer: raw signals in, PBstats features out.

    Parameters
    ----------
    fs         : sampling frequency (Hz)
    include    : feature categories (default: time, freq, nonlinear, peaks)
    preprocess : optional PBstats ``Pipeline`` run on every signal first
                 (cleaning lives inside the model → no CV leakage)
    n_jobs     : parallel workers for transform (joblib)

    Input  X : array (n_signals, n_samples) or list of 1D arrays.
    Output   : array (n_signals, n_features); names via
               ``get_feature_names_out()``. Features that cannot be computed
               for a signal are NaN — add a ``SimpleImputer`` if your model
               cannot handle NaN.
    """

    def __init__(self, fs: float = 1.0, include=None, preprocess=None, n_jobs=None):
        self.fs = fs
        self.include = include
        self.preprocess = preprocess
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        sigs = _as_signal_list(X)
        if not sigs:
            raise ValueError("X is empty")
        self.feature_names_out_ = np.array(
            list(_extract_one(sigs[0], self.fs, self.include, self.preprocess).keys()),
            dtype=object,
        )
        self.n_features_out_ = len(self.feature_names_out_)
        return self

    def transform(self, X):
        from joblib import Parallel, delayed
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "feature_names_out_")
        sigs = _as_signal_list(X)
        rows = Parallel(n_jobs=self.n_jobs)(
            delayed(_extract_one)(x, self.fs, self.include, self.preprocess) for x in sigs
        )
        names = list(self.feature_names_out_)
        return np.array([[row.get(n, np.nan) for n in names] for row in rows], dtype=float)

    def get_feature_names_out(self, input_features=None):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    def _more_tags(self):
        return {"X_types": ["2darray"], "allow_nan": False, "stateless": True}
