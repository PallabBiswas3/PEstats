# BRAIN.md — PBstats project memory

The single place to understand **what PBstats is, how it got here, what it can do, how it is built, and where it is going**. Keep it current: when you change something significant, update the relevant section and the decision log.

*Last updated: 2026-11-08 · version 0.3.0 · 149 tests*

---

## 1. What PBstats is

A Python library that takes a **raw, messy recording** (ECG, EEG, EMG, vibration, sensor logs, audio…) to **trustworthy analysis and ML-ready features**, through one chainable `Data` object:

```
load → diagnose → clean → transform / analyse → events → features → ML / report
```

Its identity — what distinguishes it from a bag of SciPy wrappers:

1. **Signal diagnosis** — `diagnose()` says what is wrong with a recording, scores it 0–100 and *prescribes* a cleaning pipeline. Accuracy is measured on PhysioNet, not assumed.
2. **Exploration becomes a pipeline** — every call is recorded; `d.to_pipeline()` replays it on a whole dataset.
3. **ML bridge** — scikit-learn transformer, feature tables, augmentation with realistic artefacts, PyTorch export.
4. **Real data in, shareable results out** — pure-Python readers (CSV, WAV, EDF, WFDB/PhysioNet) and one-file HTML reports.

**Names:** the GitHub repo is `PallabBiswas3/PEstats`; the Python package and PyPI name are `PBstats`. (Renaming the repo to `PBstats` would remove confusion — see roadmap.)

**Who it serves best:** biomedical students/researchers (ECG/HRV is the validated core), ML practitioners on time series, engineers doing condition monitoring, signal-processing learners.

---

## 2. History of changes

### v0.1.0 / v0.1.1 — original (May 2026)
Mixin-based `Data` class with cleaning, normalisation, outliers, imputation, Butterworth/notch/FIR/Savitzky-Golay filters, FFT/STFT/CWT/Hilbert, descriptive stats, hypothesis tests, regression, correlation, feature extraction (time/freq/HRV/peaks), plots, a dashboard, and a JSON-serialisable `Pipeline`. 13 tests.

**Bugs found in it** (all fixed in 0.2.0): `peak_features()`/`auto_features()` always crashed (`scipy.stats.find_peaks` does not exist); FFT amplitudes were half the true value (window gain ignored); filters and cleaning ran *across channels* on 2D data; `impute("median")` crashed on 1D; `reset()` didn't restore `fs`; resampling rounded non-integer rates; result dataclasses were defined twice; autocorrelation CI used the lag count instead of N; `plot_envelope`/`plot_stats` plotted raw instead of processed data; sample entropy took ~16 s for 1,000 samples; summaries crashed on Windows cp1252 consoles; plotting mutated global matplotlib settings. **Also:** pytest had been importing a stale *installed* copy of the package instead of the source tree.

### v0.2.0 — the "make it unique" release (2026-11-08)
- **Signal diagnosis** (`PBstats/diagnostics/quality.py`): missing data, clipping, flatlines, spikes, DC offset, drift, mains, HF noise, non-stationarity, digital signals → score, grade, issues, prescription, `auto_clean()`, `plot_quality()`.
- **Step recording** (`core/steps.py`, `@step` decorator) → `Data.steps`, `Data.to_pipeline()`.
- **ML bridge** (`ml.py`): `SignalFeatureExtractor`, `extract_features`; `windowed_features`.
- New analyses: Welch `psd`, cross-correlation (time delay), coherence, effect sizes, poly detrend, `crop`, `median_filter`, KNN imputation, Hampel and `spike` outlier methods.
- Features rebuilt as pure functions (`features/functions.py`): KD-tree sample entropy (exact, ~1000× faster), normalised permutation entropy, Higuchi FD, DFA α, Welch spectral features, Poincaré HRV.
- Multichannel support throughout; scoped plot styling; `pyproject` pytest `pythonpath`.
- Tests 13 → 114. Released to PyPI manually.

### v0.3.0 — real data, validation, automation (2026-11-08)
- **Readers** (`io.py`, pure NumPy): `from_csv` (fs from timestamps, `regularize=`), `from_wav`, `from_edf` (EDF/EDF+/BDF + annotations), `from_wfdb` (formats 16/24/32/61/80/212 + `.atr`), `from_physionet` (download + cache in `~/.cache/pbstats` or `$PBSTATS_DATA`). Bit-identical to `wfdb`/`pyedflib`.
- `Data.annotations` (DataFrame with `onset` seconds) — constructor arg, shifted by `crop()`, restored by `reset()`, copied by `channel()`.
- **Pan-Tompkins** (`features/ecg.py`) as the default R-peak detector + `match_beats` scoring.
- **ECG quality checks** — `diagnose(signal_type="ecg")`: HR plausibility, gaps, beat-morphology consistency; irregular RR is *info only*. `quality_timeline()`.
- **HRV R-R artefact rejection** (`clean_rr`, `rr_filter`, `hrv_artifact_pct`).
- **Reports** (`report.py`), **change points** (PELT) + **anomaly scores** (`events.py`), **epochs** (`epochs.py`), **forecasting** (`forecast.py`), **augmentation & DL export** (`augment.py`).
- **Benchmarks** (`benchmarks/`), **CI** (`.github/workflows/tests.yml`), **PyPI trusted publishing** (`publish.yml`), **docs site** (`mkdocs.yml`, `docs/`, `docs.yml`), `CHANGELOG.md`.
- Tests 114 → 149.

---

## 3. Decision log — why things are the way they are

Read this before "simplifying" something; most of these were learned the hard way.

| Decision | Why |
|---|---|
| Time runs along **axis 0**; 2D = `(n_samples, n_channels)` | DataFrame convention; v0.1 filters ran along axis −1 and mixed channels. |
| Domain transforms (`fft`, `hilbert`, `stft`, `wavelet`) replace `self.data`; analyses (`psd`, `describe`, `diagnose`, features, `forecast`) never do | Chaining in a transform domain was the original design; analyses must be safe to call anytime. |
| `@step` records only **outermost** calls | `normalize("zscore")` → `standardize()` must replay once, not twice. |
| Pipelines refuse names starting with `_` | Pipelines are loaded from JSON that may come from someone else. |
| Spike detector `method="spike"`: residual from a 5-sample median vs **max(noise, local slope)** | Plain Hampel flagged ECG R-peaks as spikes. The slope term protects sharp real peaks. |
| Spike noise estimate from **first differences** (MAD/√2), not the median-filter residual | The residual is exactly 0 for most samples → MAD = 0 → every wiggle looked like a spike (24/72 false alarms on near-noiseless ECG). |
| Mains needs **≥ 10 dB above background AND ≥ 0.2 % of power** | dB-only fired on noiseless signals (background = round-off) and on a square wave's 25th harmonic. If the line holds > 50 % of power it is reported as info (it may *be* the signal). |
| Signals with ≤ 16 distinct values skip clipping/flatline checks | A square wave "clips" by definition. |
| ECG irregular RR is **info**, morphology uses **median** template + **median** correlation | Record 119 (bigeminy, 444 PVCs) produced 47 % false alarms when arrhythmia counted as poor quality; now 3.6 %. |
| ECG morphology judged on a 0.5–40 Hz band-passed copy | Drift and mains have their own checks; don't double-count them as "noise". |
| `clean_rr` local median uses **mirror** padding | `nearest` padding replicated an outlier at the record edge until it became its own median (SDNN 100 ms instead of the true 41 ms). |
| HRV / feature functions always return the **full key set** (NaN when unavailable) | ML feature matrices need a fixed width across recordings. |
| Forecast model chosen by **2-fold rolling-origin validation**, best single model, refit with a divergence guard | Measured: AIC picked models with skill −1.87; 1-fold validation was noisy; averaging top-k exploded when one member diverged. |
| `report()` never calls `plt.switch_backend` | Switching closes all of the user's open figures; saving to a buffer never calls `show()`. |
| Plots use `@styled` (`rc_context`) | Never mutate the user's global matplotlib config. |
| Console output goes through `safe_print` | Box-drawing characters crash cp1252 consoles; falls back to ASCII. |
| Readers are pure Python | No new dependencies; correctness verified against the reference libraries. |
| pytest config `pythonpath = ["."]` | A stale installed copy was silently being tested instead of the source. |

---

## 4. Current capabilities

| Area | Entry points | Validation / notes |
|---|---|---|
| Loading | `Data.from_csv/wav/edf/wfdb/physionet`, `PBstats.io.read_annotations`, `BEAT_SYMBOLS` | Identical to `wfdb`/`pyedflib` (MIT-BIH 100: diff 0.0, 2,274/2,274 annotations) |
| Diagnosis | `diagnose(signal_type=None/"ecg")`, `auto_clean`, `quality_timeline`, `plot_quality(_timeline)`, `QualityReport.pipeline` | NSTDB: ECG checks flag 74/91/90/84 % of noisy windows at 12/6/0/−6 dB; 3.6 % false alarms; 0/96 false alarms on synthetic clean ECG |
| Cleaning | `remove_missing`, `impute` (incl. KNN), `remove_outliers` (zscore/iqr/mad/hampel/spike), `winsorize`, `clip`, `crop`, `remove_dc`, `detrend` (linear/constant/poly) | per channel |
| Filters | `lowpass/highpass/bandpass/bandstop` (zero-phase SOS), `notch` (+harmonics), `fir_filter`, `savgol`, `median_filter`, `resample` (rational, non-integer fs), `plot_filter_response` | clear errors for short signals |
| Transforms | `fft` (amplitude-calibrated), `psd` (Welch), `stft`, `wavelet` (log-spaced CWT), `hilbert` | |
| Statistics | `describe`, `test_normality`, `test_stationarity`, `ttest` (+Cohen's d), `mannwhitney` (+rank-biserial), `correlate`, `autocorrelation`, `cross_correlation`, `coherence`, `linear_regression` | |
| ECG | `features.ecg.pan_tompkins`, `detect_r_peaks`, `match_beats`, `hrv_features` | MIT-BIH 48 records: Se 99.57 %, +P 99.11 % (simple: 88.6/93.7) |
| Features | `time/freq/nonlinear/peak/hrv_features`, `auto_features`, `windowed_features`, `features.functions.extract` | entropies/FD/DFA checked against theory and a naive reference |
| Events | `detect_change_points` (PELT, exact), `anomaly_scores` (Isolation Forest), `plot_events` | PELT matches exhaustive DP; 1M samples in seconds via block grid |
| Epochs | `epochs()` → `Epochs` (`average/median/std`, `["label"]`, `to_numpy("NCT")`, `plot`) | |
| Forecasting | `forecast(steps/horizon, holdout, order, select, n_jobs)`, `plot_forecast` | metrics incl. coverage and skill vs naive |
| ML / DL | `SignalFeatureExtractor`, `extract_features`, `Augmenter`, `make_windows`, `to_torch_dataset` | augmented artefacts are detected by `diagnose()` |
| Output | `report()` HTML, `dashboard()`, ~15 plot methods, `to_pipeline()`, `Pipeline.save/load/run_batch` | |

---

## 5. Architecture

```
PBstats/
  __init__.py        public API; lazy import of ml (scikit-learn) names
  core/
    data.py          Data = all mixins; constructor, readers, reset/copy/channel, to_pipeline
    steps.py         @step decorator → Data._steps
    pipeline.py      Pipeline, FunctionalPipeline (JSON, batch, timing)
    base.py          result dataclasses (FFT/PSD/Hilbert/STFT/Wavelet)
    utils.py         require_1d, per_channel, check_length, interp_nans, safe_print
  io.py              readers + PhysioNet download
  diagnostics/quality.py   checks, QualityReport, QualityMixin (diagnose/auto_clean/quality_timeline)
  preprocessing/     cleaning, imputation, outliers (detect_outliers), normalization, filters
  transforms/        fft (+psd), hilbert, stft, wavelet
  statistics/        descriptive, hypothesis, regression, correlation (+results in base.py)
  features/
    functions.py     pure feature functions + extract()   ← reusable without Data
    ecg.py           Pan-Tompkins, match_beats
    extractor.py     FeatureMixin, FeatureVector
  events.py  epochs.py  forecast.py  augment.py  report.py  ml.py
  visualization/     base (style, save_or_show), plots (functions), spectrum (VisualizationMixin)
tests/               pytest; conftest.synth_ecg; fileformats.py = spec writers for EDF/WFDB
benchmarks/          rpeaks_mitdb.py, quality_nstdb.py, results/
docs/ + mkdocs.yml   documentation site;  .github/workflows: tests, publish, docs
```

### Conventions for adding a method to `Data`
1. Put it in the right mixin (or a new mixin added to `Data`'s bases in `core/data.py`).
2. Decorate with `@step` if it should be replayable; return `self` if chainable.
3. Use `self._require_fs("name")`, `require_1d(...)` / `per_channel(...)` from `core/utils.py`.
4. Store results on `self.<name>_result` (cleared automatically by `reset()`); call `self._record("...")`.
5. Pure computations go in a function (testable without `Data`); the method wraps it.
6. Plots: a `@styled` function in `visualization/plots.py` ending with `save_or_show(fig, save_path)`, plus a thin method in `spectrum.py`.
7. Printing: `safe_print`. Warnings: `warnings.warn(..., stacklevel=2)`.
8. Tests with known ground truth; avoid shared module-level RNGs (each test seeds its own).
9. Update docs (`docs/guides`, API pages), `CHANGELOG.md`, and this file.

---

## 6. Working on the project

```bash
pip install -e ".[dev]"                     # editable install (avoid stale copies)
python -m pytest                            # 149 tests, ~1 min
ruff check PBstats tests benchmarks examples --select F,E9,B --ignore B905,B008
python examples/demo.py                     # end-to-end story
python benchmarks/rpeaks_mitdb.py           # ~100 MB PhysioNet download once
python benchmarks/quality_nstdb.py
pip install -e ".[docs]" && mkdocs build --strict
```

**Release process:** bump `version` in `pyproject.toml` **and** `PBstats/__init__.py` → update `CHANGELOG.md` → push → GitHub Release with tag `vX.Y.Z` → `publish.yml` tests, builds and uploads to PyPI (fails if tag ≠ version).

**Pending one-time setup (as of 0.3.0):**
- [ ] PyPI → PBstats → Publishing → add GitHub publisher (`PallabBiswas3` / `PEstats` / `publish.yml` / env `pypi`)
- [ ] GitHub → Settings → Pages → Source: GitHub Actions (docs site)
- [ ] Create GitHub Release `v0.3.0` to publish it

---

## 7. Known limitations

- Research tool, **not** certified medical software.
- Non-ECG diagnosis is heuristic and benchmarked only indirectly; only ECG has a real-data quality benchmark.
- ECG checks need ≥ 3 s and fs ≥ 50 Hz; very noisy records (e.g. MIT-BIH 105) are diluted in whole-record diagnosis — use `quality_timeline`.
- Pan-Tompkins is weakest on ventricular flutter (MIT-BIH 207), swapped leads (114), large P/T waves (108).
- Forecasting is classical ARIMA; full grid search takes seconds to tens of seconds.
- Transforms are single-channel (use `d.channel(i)`); EDF channels must share a sampling rate per `Data`.
- No streaming / real-time processing; whole recordings are held in memory.
- WFDB multi-segment records are not supported.
- Material for MkDocs reaches end of life May 2027 (successor: Zensical).

---

## 8. Roadmap

**Next (high value)**
- Signal-quality benchmarks for **EEG** (e.g. artefact-annotated datasets) and **PPG**, mirroring the NSTDB study.
- `diagnose(signal_type="eeg")` (eye blinks, muscle, flat channels, bad-channel detection across a montage) and `"ppg"`.
- Multichannel diagnosis and reports (per-channel table + bad-channel summary).
- Rename the GitHub repo to `PBstats`; add a citation file (`CITATION.cff`) and a Zenodo DOI.

**Then**
- More detectors: Hamilton / wavelet QRS, PPG pulse detection, EMG onset detection.
- Arrhythmia-aware HRV (ectopic beat interpolation instead of rejection).
- Faster forecasting (parallel by default, smaller grid, or `statsforecast` optional backend); seasonal search.
- Streaming / chunked processing for long recordings that don't fit in memory.
- Type hints throughout + `py.typed`; property-based tests for readers.
- Example notebooks / gallery in the docs (sleep EEG, arrhythmia ML, machine vibration).

**Ideas**
- Interactive HTML report (zoomable plots).
- Pretrained baseline models (e.g. noise classifier trained with `Augmenter`).
- CLI: `pbstats diagnose file.edf --report out.html`.
