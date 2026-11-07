# ECG & HRV

## R-peak detection

PBstats implements the Pan-Tompkins QRS detector (Pan & Tompkins, *IEEE TBME* 1985): 5–15 Hz band-pass, derivative, squaring, 150 ms moving-window integration, adaptive signal/noise thresholds, search-back for missed beats and T-wave discrimination. It works on either polarity and needs no pre-filtering.

```python
from PBstats.features.ecg import detect_r_peaks, match_beats

r = detect_r_peaks(d.data, d.fs)                    # Pan-Tompkins (default)
r_simple = detect_r_peaks(d.data, d.fs, method="simple")
match_beats(reference_samples, r, d.fs)             # Se / +P / F1 at ±150 ms
```

On all 48 MIT-BIH Arrhythmia records it reaches **99.57 % sensitivity and 99.11 % positive predictivity** (109,494 beats) — see [Benchmarks](../benchmarks.md).

## HRV features

```python
d = Data.from_physionet("mitdb/100", channels="MLII")
d.hrv_features().feature_vector.features
```

| Feature | Meaning |
|---|---|
| `hrv_mean_rr_ms`, `hrv_mean_hr_bpm`, `hrv_sdhr_bpm` | average interval / rate |
| `hrv_sdnn_ms` | overall variability |
| `hrv_rmssd_ms`, `hrv_pnn50_pct` | short-term (vagal) variability |
| `hrv_sd1_ms`, `hrv_sd2_ms`, `hrv_sd1_sd2` | Poincaré plot |
| `hrv_tri_index` | triangular index (1/128 s bins) |
| `hrv_lf_power`, `hrv_hf_power`, `hrv_lf_hf_ratio` | spectral (≥ 20 beats) |
| `hrv_artifact_pct` | share of R-R intervals rejected |

R-R intervals are cleaned before computing metrics: values outside 250–2500 ms, or deviating more than 30 % from the median of the surrounding 11 beats, are rejected (`rr_filter=None` keeps everything). A single false beat can otherwise double SDNN. The full key set is always returned (NaN where not computable) so feature tables keep a fixed width.

## ECG signal quality

```python
d.diagnose(signal_type="ecg")
```

adds heart-rate plausibility and beat-morphology consistency checks — see [Signal diagnosis](diagnosis.md).

## Average beats

```python
ep = d.epochs(["N", "V"], tmin=-0.25, tmax=0.45)    # from PhysioNet beat labels
normal, pvc = ep["N"].average(), ep["V"].average()
```
