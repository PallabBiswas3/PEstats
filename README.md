# PBstats

[![tests](https://github.com/PallabBiswas3/PEstats/actions/workflows/tests.yml/badge.svg)](https://github.com/PallabBiswas3/PEstats/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/PBstats.svg)](https://pypi.org/project/PBstats/)
[![Python](https://img.shields.io/pypi/pyversions/PBstats.svg)](https://pypi.org/project/PBstats/)
[![docs](https://img.shields.io/badge/docs-online-5B4FCF.svg)](https://pallabbiswas3.github.io/PEstats/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**From a raw, messy recording to ML-ready features — with a diagnosis, a reproducible recipe, and a model at the end.**

PBstats is a signal-aware analysis library for time series and biosignals (ECG, EEG, EMG, vibration, sensors, spectroscopy). One chainable `Data` object takes you through loading, diagnosis, cleaning, transforms, statistics, events, features and machine learning. What makes it different:

1. **It tells you what's wrong with your signal.** `diagnose()` finds gaps, dropout, clipping, spikes, drift, mains interference and noise — and for ECG, implausible rhythms and inconsistent beat shapes. It grades the recording 0–100 and *prescribes* a cleaning pipeline. Accuracy is [measured on PhysioNet](#benchmarks), not assumed.
2. **Your exploration becomes your pipeline.** Every chained call is recorded, so `d.to_pipeline()` turns what you tried interactively into a saveable pipeline you can run on a whole dataset.
3. **It plugs into machine learning.** A scikit-learn transformer for features, plus augmentation with realistic artefacts and windowing for deep learning.
4. **It reads real data and writes shareable reports.** CSV, WAV, EDF/EDF+/BDF and PhysioNet — pure Python, verified identical to the reference libraries — and one-file HTML reports.

📖 **Documentation: <https://pallabbiswas3.github.io/PEstats/>**

## Installation

```bash
pip install PBstats
pip install "PBstats[dl]"     # + PyTorch dataset helpers
```

## 60-second tour

```python
from PBstats import Data

d = Data.from_physionet("mitdb/105", channels="MLII")      # real ECG, downloaded once and cached
d.diagnose(signal_type="ecg", verbose=True)                # what's wrong with it?
d.auto_clean()                                             # apply the prescribed fixes
d.hrv_features().feature_vector.to_series()                # Pan-Tompkins beats → HRV
d.report("mitdb_105.html", signal_type="ecg")              # one-file shareable report
```

```text
Signal diagnosis — 'patient_001'  (15000 samples, 30.0 s @ 500 Hz)
  Quality  █████░░░░░░░░░░░░░░░  26/100  grade F
  ✖ [critical] Power-line interference at 50 Hz (42 dB above the local background).
  ▲ [warning ] 60 missing samples (0.4%) in 1 gap(s) — filters and FFTs fail on NaN.
  ▲ [warning ] 6 impulse spike sample(s) (> 10× both noise level and local slope) — ...
  ▲ [warning ] 72% of the power is below 0.5 Hz — baseline wander/drift. ...
  Prescription (apply with .auto_clean() or report.pipeline):
    1. remove_missing(strategy='interpolate')
    2. remove_outliers(method='spike', threshold=10.0, window=5)
    3. highpass(cutoff=0.5)
    4. notch(freq=50.0, harmonics=False)
```

## Explore once, replay everywhere

```python
d = Data(x, fs=500).remove_missing().bandpass(0.5, 40).notch(50).standardize()

pipe = d.to_pipeline("ecg_clean")          # exactly what you did, with every argument
pipe.save("ecg_clean.json")
cleaned = Pipeline.load("ecg_clean.json").run_batch(recordings, on_error="warn")
```

## Events, epochs and forecasting

```python
d.detect_change_points()                   # exact PELT segmentation → d.change_point_result
d.anomaly_scores(window=5)                 # Isolation Forest over windowed features
ep = d.epochs("N", tmin=-0.25, tmax=0.45)  # cut around annotated events
ep.average().plot()                        # ERP / average heartbeat
d.forecast(horizon=10, holdout=100)        # ARIMA, honest held-out metrics
d.quality_timeline(window=10)              # where is a long recording bad?
```

## Machine learning

```python
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from PBstats import Augmenter, Pipeline, SignalFeatureExtractor, make_windows, to_torch_dataset

model = make_pipeline(                                  # cleaning lives inside the model,
    SignalFeatureExtractor(fs=250, include=["time", "freq", "nonlinear"],   # so CV stays honest
                           preprocess=Pipeline("clean").add("bandpass", low=1, high=40)),
    RandomForestClassifier(),
)

X, y = make_windows(recordings, window=10, step=5, labels=labels)       # (N, C, T) float32
aug = Augmenter(fs=250, powerline=0.3, wander=0.3, spikes=0.2, seed=0)  # artefacts diagnose() knows
ds = to_torch_dataset(X, y, augmenter=aug)                              # on-the-fly augmentation
```

## Benchmarks

Reproducible with the scripts in [`benchmarks/`](benchmarks/) on public PhysioNet data.

**R-peak detection, MIT-BIH Arrhythmia Database** (48 records, 109,494 beats, ±150 ms):

| Detector | Sensitivity | +Predictivity | F1 |
|---|---|---|---|
| Pan-Tompkins (default) | **99.57 %** | **99.11 %** | **99.34 %** |
| Simple threshold | 88.62 % | 93.67 % | 91.08 % |

**Signal quality, MIT-BIH Noise Stress Test Database** (10-s windows with known noise; `signal_type="ecg"`):

| Noise SNR | 24 dB | 18 dB | 12 dB | 6 dB | 0 dB | −6 dB | False alarms (clean ECG) |
|---|---|---|---|---|---|---|---|
| Noisy windows flagged | 5 % | 17 % | 74 % | 91 % | 90 % | 84 % | **3.6 %** |

![Quality over time on NSTDB record 118e06 — flagged blocks match the known noise segments](docs/img/nstdb_timeline.png)

Detection rises exactly where noise starts to break beat detection (Pan-Tompkins F1 is still 99.97 % at 18 dB, 96.9 % at 12 dB). Details and caveats: [Benchmarks](https://pallabbiswas3.github.io/PEstats/benchmarks/).

## What's inside

| Area | Highlights |
|---|---|
| **Loading** | `from_csv` (fs from timestamps), `from_wav`, `from_edf` (EDF/EDF+/BDF + annotations), `from_wfdb`, `from_physionet` |
| **Diagnostics** | `diagnose` (+ ECG checks), `auto_clean`, `quality_timeline`, `plot_quality`, `report` |
| **Cleaning** | `remove_missing`, `impute` (incl. KNN), `remove_outliers` (zscore / iqr / mad / hampel / spike), `winsorize`, `clip`, `crop`, `remove_dc`, `detrend` |
| **Filters** | zero-phase Butterworth low/high/band-pass/stop, `notch` (+ harmonics), FIR, Savitzky-Golay, median, polyphase `resample` |
| **Transforms** | amplitude-calibrated `fft`, Welch `psd`, `stft`, `wavelet`, `hilbert` |
| **Statistics** | `describe`, normality & stationarity tests, t-test with Cohen's d, Mann-Whitney with rank-biserial r, correlation, autocorrelation, cross-correlation, coherence |
| **Events** | `detect_change_points` (PELT), `anomaly_scores`, `epochs`, `forecast` (ARIMA) |
| **Features** | time + Hjorth; spectral; sample/permutation entropy, Higuchi FD, DFA; peaks; HRV with Pan-Tompkins, R-R artefact rejection, Poincaré, LF/HF |
| **ML / DL** | `SignalFeatureExtractor`, `extract_features`, `windowed_features`, `Augmenter`, `make_windows`, `to_torch_dataset` |

Multichannel data — a 2D array `(n_samples, n_channels)` or a DataFrame — is handled per channel throughout.

## Development

```bash
git clone https://github.com/PallabBiswas3/PEstats.git && cd PEstats
pip install -e ".[dev]"
python -m pytest                    # 149 tests
python examples/demo.py             # messy ECG → diagnosis → pipeline → features → classifier
python benchmarks/rpeaks_mitdb.py   # reproduce the benchmarks
```

Releases are published to PyPI automatically when a GitHub Release is created (see [CHANGELOG](CHANGELOG.md)).

---
Developed with passion for the signal processing community.
