# Changelog

All notable changes to PBstats. Versions follow [semantic versioning](https://semver.org/).

## 0.3.0 — 2026-11-08

### Added
- **File readers** (pure Python, no new dependencies): `Data.from_csv` (sampling rate inferred from timestamps, optional regularisation), `from_wav`, `from_edf` (EDF, EDF+ with annotations, BDF), `from_wfdb` (PhysioNet formats 16/24/32/61/80/212 plus `.atr` annotations) and `from_physionet` (download + cache). Verified identical to the reference `wfdb` / `pyedflib` libraries.
- `Data.annotations`: events with `onset` in seconds, kept aligned by `crop()` and restored by `reset()`.
- **Pan-Tompkins R-peak detector**, now the default for HRV — 99.57 % Se / 99.11 % +P on all 48 MIT-BIH records (was 88.6 % / 93.7 %). `match_beats()` scores detections.
- **ECG signal-quality checks**: `diagnose(signal_type="ecg")` adds heart-rate plausibility and beat-morphology consistency (Orphanidou et al. 2015). Validated on the MIT-BIH Noise Stress Test Database: 84–91 % of noisy windows flagged at ≤ 6 dB SNR, 3.6 % false alarms on clean ECG.
- `quality_timeline()` and `plot_quality_timeline()` — quality per window.
- **HTML reports**: `d.report("file.html")`, self-contained with embedded figures.
- **Change points** (`detect_change_points`, exact PELT) and **anomaly scores** (`anomaly_scores`, Isolation Forest on windowed features), with `plot_events()`.
- **Epochs**: `d.epochs(events, tmin, tmax, baseline, reject)` → `Epochs` with `average()`, `median()`, label selection, `to_numpy()`, `plot()`.
- **Forecasting**: `d.forecast()` with ARIMA, model choice by rolling-origin validation, held-out metrics including skill versus a naive forecast, `plot_forecast()`.
- **Augmentation & deep learning**: `Augmenter` (noise, mains, wander, spikes, dropout, gain, shift, stretch, inversion), `make_windows`, `to_torch_dataset`.
- HRV: R-R artefact rejection (`rr_filter`) and `hrv_artifact_pct`.
- Benchmark scripts (`benchmarks/`), GitHub Actions for tests (3 OS × Python 3.10–3.13), PyPI trusted publishing, and a documentation site.

### Fixed
- HRV metrics inflated by a single false beat at the end of a recording.
- `match_beats` returned NaN when nothing was detected.
- Deprecated Matplotlib `boxplot(vert=...)` usage.

## 0.2.0 — 2026-11-08

### Added
- Signal diagnosis (`diagnose`, `auto_clean`, `plot_quality`) with a 0–100 score and a prescribed cleaning pipeline.
- Step recording and `Data.to_pipeline()`; `Pipeline` composition, `copy=` runs, JSON-safety.
- `SignalFeatureExtractor` (scikit-learn), `extract_features`, `windowed_features`.
- Welch `psd`, `cross_correlation`, `coherence`, effect sizes, polynomial detrending, `crop`, `median_filter`, KNN imputation, Hampel and spike outlier methods.
- Nonlinear features (sample/permutation entropy, Higuchi FD, DFA) and Poincaré HRV.
- Multichannel support throughout.

### Fixed
- `peak_features()` / `auto_features()` always crashed.
- FFT amplitudes were half the true value.
- Filters and cleaning operated across channels on 2D data; plus several smaller bugs.

## 0.1.1 — 2026-05-25
- Filters and feature extraction modules.

## 0.1.0 — 2026-05-25
- First release.
