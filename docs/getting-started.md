# Getting started

## The `Data` object

Everything starts with a `Data`: your samples, the sampling rate, a label, and a record of everything you do to it.

```python
import numpy as np
from PBstats import Data

fs = 500
t = np.arange(10 * fs) / fs
signal = np.sin(2 * np.pi * 8 * t) + 0.3 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(len(t))

d = Data(signal, fs=fs, label="demo")
```

- 1D arrays are one channel; 2D arrays are `(n_samples, n_channels)`. DataFrame columns become channel names.
- Processing methods change `d.data` and return `d`, so they chain.
- `d.reset()` undoes everything; `d.copy()` branches.

## Chain operations

```python
d = (Data(signal, fs=fs)
     .notch(50)              # remove mains hum
     .bandpass(1, 40)        # zero-phase Butterworth
     .standardize())

d.psd().plot_psd()           # Welch spectrum (non-destructive)
d.describe().plot_stats()    # descriptive statistics
```

Transforms that change domain — `fft()`, `hilbert()`, `stft()`, `wavelet()` — replace `d.data` with their output (the full result is in `d.fft_result`, etc.). Analyses such as `psd()`, `describe()`, `diagnose()` and the feature methods never modify the data.

## Diagnose before you analyse

```python
d = Data(signal, fs=fs).diagnose(verbose=True)
d.quality_report.score       # 0–100
d.quality_report.fixes       # [('notch', {'freq': 50.0, 'harmonics': False})]
d.auto_clean()               # apply them
```

See [Signal diagnosis](guides/diagnosis.md).

## Extract features

```python
d.auto_features()            # time, frequency, nonlinear, peak features
d.feature_vector.to_series() # ~45 named features
```

## Keep it reproducible

```python
pipe = d.to_pipeline("my_cleaning")
pipe.save("my_cleaning.json")
```

See [Pipelines](guides/pipelines.md).
