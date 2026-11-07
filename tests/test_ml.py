import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from PBstats import Pipeline, SignalFeatureExtractor, extract_features

FS = 100


def _dataset(n_per_class=20, seed=0):
    """Two classes: 5 Hz vs 12 Hz oscillations in noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(4 * FS) / FS
    X, y = [], []
    for label, f in ((0, 5.0), (1, 12.0)):
        for _ in range(n_per_class):
            X.append(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) + 0.8 * rng.standard_normal(len(t)))
            y.append(label)
    return np.array(X), np.array(y)


def test_extract_features_table():
    X, y = _dataset(5)
    df = extract_features(X, fs=FS, labels=y, include=["time", "freq"])
    assert df.shape[0] == 10 and df.columns[0] == "label"
    assert not df.drop(columns="label").isna().any().any()


def test_extract_features_ragged_and_parallel():
    sigs = [np.random.default_rng(i).standard_normal(200 + 50 * i) for i in range(4)]
    a = extract_features(sigs, fs=FS, include=["time"], n_jobs=1)
    b = extract_features(sigs, fs=FS, include=["time"], n_jobs=2)
    assert a.equals(b)


def test_extract_features_on_error_nan():
    sigs = [np.ones(300), np.array([1.0, np.nan] * 150)]
    df = extract_features(sigs, fs=FS, include=["time"], on_error="nan")
    assert df.iloc[1].isna().all()
    with pytest.raises(ValueError):
        extract_features(sigs, fs=FS, include=["time"])


def test_transformer_in_sklearn_pipeline():
    X, y = _dataset()
    clean = Pipeline("clean").add("bandpass", low=1, high=30).add("standardize")
    model = make_pipeline(
        SignalFeatureExtractor(fs=FS, include=["time", "freq"], preprocess=clean),
        RandomForestClassifier(n_estimators=50, random_state=0),
    )
    scores = cross_val_score(model, X, y, cv=4)
    assert scores.mean() > 0.9


def test_transformer_contract():
    X, _ = _dataset(3)
    ext = SignalFeatureExtractor(fs=FS, include=["time"]).fit(X)
    out = ext.transform(X)
    names = ext.get_feature_names_out()
    assert out.shape == (6, len(names)) and names[0] == "td_mean"
    assert clone(ext).get_params()["fs"] == FS
    restored = pickle.loads(pickle.dumps(ext))
    assert np.allclose(restored.transform(X), out)


def test_transformer_respects_resample_in_preprocess():
    X, _ = _dataset(2)
    pipe = Pipeline("down").add("resample", target_fs=50)
    out = SignalFeatureExtractor(fs=FS, include=["freq"], preprocess=pipe).fit_transform(X)
    assert out.shape[0] == 4 and np.isfinite(out).all()
