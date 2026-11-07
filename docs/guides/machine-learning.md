# Machine learning

## Feature tables

```python
from PBstats import extract_features

df = extract_features(signals, fs=250, labels=y, include=["time", "freq", "nonlinear"], n_jobs=-1)
```

Categories: `time` (amplitude, shape, Hjorth), `freq` (Welch spectral shape, edges, band powers), `nonlinear` (sample entropy, permutation entropy, Higuchi fractal dimension, DFA α), `peaks`, `hrv`. For one long recording, `d.windowed_features(window=30, step=10)` gives a feature time series.

## scikit-learn

`SignalFeatureExtractor` takes raw signals and outputs features, so cleaning lives *inside* the model and cross-validation stays honest:

```python
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from PBstats import Pipeline, SignalFeatureExtractor

model = make_pipeline(
    SignalFeatureExtractor(fs=250, include=["time", "freq"],
                           preprocess=Pipeline("clean").add("bandpass", low=1, high=40)),
    RandomForestClassifier(),
)
cross_val_score(model, X_signals, y, cv=5)
```

## Deep learning

```python
from PBstats import Augmenter, make_windows, to_torch_dataset

X, y = make_windows(recordings, window=10, step=5, labels=labels)   # (N, C, T) float32
aug = Augmenter(fs=250, powerline=0.3, wander=0.3, spikes=0.2, dropout=0.1, seed=0)
X_aug, y_aug = aug.expand(X, y, copies=4)                           # offline
ds = to_torch_dataset(X, y, augmenter=aug)                          # on the fly
```

The augmentations mirror the artefacts `diagnose()` detects — noise, mains hum, baseline wander, spikes, dropouts, gain changes, timing jitter, inverted leads — so models learn to cope with what real recordings contain. A window corrupted by `Augmenter(powerline=1, wander=1, spikes=1)` is diagnosed as exactly `mains`, `drift` and `spikes`.
