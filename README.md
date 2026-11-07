# PBstats

**From a raw, messy recording to ML-ready features — with a diagnosis, a reproducible recipe, and a scikit-learn model at the end.**

PBstats is a signal-aware analysis framework for time series and biosignals (ECG, EEG, EMG, vibration, spectroscopy, sensors). It gives you a fluent, chainable API for cleaning, filtering, transforming, testing and featurising signals. Three things make it different from a bag of SciPy wrappers:

1. **It tells you what's wrong with your signal.** `diagnose()` checks for gaps, sensor dropout, clipping, impulse spikes, baseline drift, power-line interference, broadband noise and non-stationarity. It then grades the recording 0–100 and *prescribes* an ordered cleaning pipeline. `auto_clean()` applies it.
2. **Your exploration becomes your pipeline.** Every chained call is recorded in a structured form, so `d.to_pipeline()` turns what you just tried interactively into a saveable, replayable `Pipeline`. Then you can run it on a whole dataset.
3. **It plugs straight into machine learning.** `SignalFeatureExtractor` is a scikit-learn transformer: raw signals in, ~50 time, spectral, nonlinear, peak and HRV features out. Cleaning happens *inside* the model, so cross-validation stays honest.

## Installation

```bash
git clone https://github.com/PallabBiswas3/PEstats.git
cd PEstats
pip install -e .          # editable install: your edits take effect immediately
```

## 60-second tour

```python
from PBstats import Data

d = Data(raw_ecg, fs=500, label="patient_001")

d.diagnose(verbose=True)
```
```
Signal diagnosis — 'patient_001'  (15000 samples, 30.0 s @ 500 Hz)
────────────────────────────────────────────────────────────────
  Quality  █████░░░░░░░░░░░░░░░  26/100  grade F
────────────────────────────────────────────────────────────────
  ✖ [critical] Power-line interference at 50 Hz (42 dB above the local background).
  ▲ [warning ] 60 missing samples (0.4%) in 1 gap(s) — filters and FFTs fail on NaN.
  ▲ [warning ] 6 impulse spike sample(s) (> 10× both noise level and local slope) — ...
  ▲ [warning ] 72% of the power is below 0.5 Hz — baseline wander/drift. ...
  • [info    ] Large DC offset (|mean| = 2.8× std) — dominates the 0 Hz bin of spectra.
────────────────────────────────────────────────────────────────
  Prescription (apply with .auto_clean() or report.pipeline):
    1. remove_missing(strategy='interpolate')
    2. remove_outliers(method='spike', threshold=10.0, window=5)
    3. highpass(cutoff=0.5)
    4. notch(freq=50.0, harmonics=False)
```
```python
d.plot_quality()                      # signal with problem regions highlighted + PSD
d.auto_clean()                        # 26/100 → 100/100
d.auto_features(include=["time", "freq", "nonlinear", "hrv"])
d.feature_vector.to_series()          # ~50 named features
```

## Explore once, replay everywhere

```python
d = (Data(x, fs=500)
     .remove_missing()
     .bandpass(0.5, 40)
     .notch(50)
     .standardize())

pipe = d.to_pipeline("ecg_clean")     # exactly what you did, with every argument
pipe.save("ecg_clean.json")

cleaned = Pipeline.load("ecg_clean.json").run_batch(
    [Data(s, fs=500, label=f"rec_{i}") for i, s in enumerate(dataset)],
    on_error="warn",
)
```

Pipelines compose (`clean + features`), refuse to call private methods when loaded from JSON, and can run on a copy (`pipe.run(d, copy=True)`) to leave the input untouched.

## Machine learning

```python
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from PBstats import Pipeline, SignalFeatureExtractor

model = make_pipeline(
    SignalFeatureExtractor(
        fs=250,
        include=["time", "freq", "nonlinear"],
        preprocess=Pipeline("clean").add("bandpass", low=1, high=40),
        n_jobs=-1,
    ),
    RandomForestClassifier(),
)
cross_val_score(model, X_signals, y, cv=5)    # X: (n_signals, n_samples) or ragged list
```

Prefer a table? `extract_features(signals, fs, labels=y)` returns a DataFrame. For long recordings, `d.windowed_features(window=30, step=10)` turns one signal into a feature *time series*. That's the standard input for sleep staging, activity recognition, seizure detection and fault detection.

## What's inside

| Area | Highlights |
|---|---|
| **Diagnostics** | `diagnose`, `auto_clean`, `plot_quality` — 0–100 score, issue list, prescribed `Pipeline` |
| **Cleaning** | `remove_missing` (interpolate / ffill / bfill / mean / median / drop), `impute` (incl. KNN across channels), `remove_outliers` (zscore / iqr / mad / hampel / **spike**), `winsorize`, `clip`, `crop`, `remove_dc`, `detrend` (linear / poly) |
| **Filters** | zero-phase Butterworth `lowpass` / `highpass` / `bandpass` / `bandstop`, `notch` (+ harmonics), linear-phase `fir_filter`, `savgol`, `median_filter`, polyphase `resample` (non-integer rates), `plot_filter_response` |
| **Transforms** | amplitude-calibrated `fft`, Welch `psd`, `stft`, `wavelet` (CWT, log-spaced), `hilbert` (envelope, phase, inst. frequency) |
| **Statistics** | `describe` (incl. RMS, MAD, SNR), normality & stationarity tests, Welch t-test with Cohen's d, Mann-Whitney with rank-biserial r, `correlate`, FFT `autocorrelation`, `cross_correlation` (time-delay estimation), `coherence` |
| **Features** | time domain + Hjorth; Welch spectral shape, edges, EEG band powers; sample entropy (KD-tree), permutation entropy, Higuchi fractal dimension, DFA α; peak statistics; HRV (SDNN, RMSSD, pNN50, Poincaré SD1/SD2, LF/HF) with RR-artefact rejection |
| **Visualisation** | signal, spectrum, PSD with bands, spectrogram, scalogram, envelope, stats panel, ACF, correlation matrix, quality report, 6-panel `dashboard` — scoped styling that never touches your global matplotlib settings |

**Multichannel:** pass a 2D array `(n_samples, n_channels)` or a DataFrame (columns = channels). Cleaning, normalisation and filters work per channel along time. Features are prefixed with the channel name (`Fz__td_rms`). Use `d.channel("Fz")` for single-channel transforms.

## Design notes

- **Every method returns `self`** for chaining. Results land on attributes (`fft_result`, `psd_result`, `quality_report`, `feature_vector`, …).
- **Transforms that change the domain** (`fft`, `hilbert`, `stft`, `wavelet`) replace `self.data` with their output so a chain can continue. Analysis methods (`psd`, `describe`, `diagnose`, features) never modify the data.
- **`reset()`** restores the original data *and* sampling rate, and clears all history and results. **`copy()`** gives an independent branch.
- **Diagnosis is heuristic.** Thresholds are documented constants in `PBstats/diagnostics/quality.py`. They were tuned on biosignals and validated for zero false alarms on clean synthetic ECG at 250–1000 Hz. If slow content *is* your signal (e.g. respiration), pass `drift_cutoff=None`.

## Demo and tests

```bash
python examples/demo.py          # messy ECG → diagnosis → pipeline → features → classifier
python -m pytest                 # 114 tests
```

## Project structure

```
PBstats/
  core/           Data, Pipeline, step recording, result containers
  diagnostics/    signal-quality diagnosis and auto-cleaning
  preprocessing/  cleaning, imputation, outliers, normalisation, filters
  transforms/     FFT, Welch PSD, STFT, CWT, Hilbert
  statistics/     descriptive, hypothesis tests, regression, correlation
  features/       pure feature functions + Data feature methods
  visualization/  plotting
  ml.py           scikit-learn transformer and batch feature extraction
```

---
Developed with passion for the signal processing community.
