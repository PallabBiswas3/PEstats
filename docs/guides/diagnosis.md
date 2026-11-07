# Signal diagnosis

`diagnose()` inspects a recording for the problems that silently ruin analysis, grades it 0–100, explains each finding and prescribes a cleaning pipeline.

```python
d = Data(raw, fs=500, label="patient_001").diagnose(signal_type="ecg", verbose=True)
```

```text
Signal diagnosis — 'patient_001'  (15000 samples, 30.0 s @ 500 Hz)
  Quality  █████░░░░░░░░░░░░░░░  26/100  grade F
  ✖ [critical] Power-line interference at 50 Hz (42 dB above the local background).
  ▲ [warning ] 60 missing samples (0.4%) in 1 gap(s) — filters and FFTs fail on NaN.
  ▲ [warning ] 6 impulse spike sample(s) ...
  ▲ [warning ] 72% of the power is below 0.5 Hz — baseline wander/drift.
  Prescription (apply with .auto_clean() or report.pipeline):
    1. remove_missing(strategy='interpolate')
    2. remove_outliers(method='spike', threshold=10.0, window=5)
    3. highpass(cutoff=0.5)
    4. notch(freq=50.0, harmonics=False)
```

## Checks

| Code | Detects | Prescribed fix |
|---|---|---|
| `missing` | NaN gaps | `remove_missing("interpolate")` |
| `clipping` | saturation at min/max | — (information lost) |
| `flatline` | sensor dropout | — (crop it out) |
| `spikes` | impulse artefacts, judged against noise *and* local slope so sharp real peaks survive | `remove_outliers("spike")` |
| `dc_offset` | large offset (info) | `remove_dc()` |
| `drift` | power below 0.5 Hz (tunable `drift_cutoff`) | `highpass(0.5)` |
| `trend` | linear trend when the record is too short for a spectral check | `detrend()` |
| `mains` | 50/60 Hz line ≥ 10 dB above background (and ≥ 0.2 % of power) | `notch(f0)` |
| `hf_noise` | broadband noise near Nyquist (info) | — |
| `nonstationary` | amplitude varying > 5× across the record (info) | — |
| `digital` | ≤ 16 distinct values (info; disables clipping/flatline checks) | — |

With `signal_type="ecg"`, heartbeat checks after Orphanidou et al. (2015) are added:

| Code | Detects |
|---|---|
| `ecg_beats` | fewer than 3 beats — not a usable ECG |
| `ecg_rhythm` | mean HR outside 40–180 bpm or > 3 s without a beat |
| `ecg_morphology` | median beat-to-template correlation < 0.66 — in-band noise (muscle, motion) that filters cannot remove |
| `ecg_irregular` | irregular RR intervals (info only: ectopics/arrhythmia are clinical content, not noise) |

Scores: each critical issue costs 35 points, a warning 12, info 3. `report.ok` is true when there are no warnings or criticals.

## Where is it bad? — `quality_timeline`

Long recordings are rarely uniformly good or bad:

```python
tl = d.quality_timeline(window=10, signal_type="ecg")   # one row per 10-s window
d.plot_quality_timeline()
clean_parts = tl[tl.ok]
```

On the noise-stress record `118e06` this flags 95 % of the noisy windows and 5 % of the clean ones.

## Applying the prescription

```python
d.auto_clean()                         # diagnose + apply; before/after reports kept
d.quality_report_after.score

rx = d.quality_report.pipeline         # or freeze it for a whole dataset
rx.save("clean.json")
```

!!! warning "These are heuristics"
    Thresholds are documented constants in `PBstats/diagnostics/quality.py`, tuned on biosignals. Their accuracy on real data is measured in [Benchmarks](../benchmarks.md). If slow content *is* your signal (e.g. respiration), pass `drift_cutoff=None`.
