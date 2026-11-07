# Benchmarks

Measured on public PhysioNet databases with the scripts in [`benchmarks/`](https://github.com/PallabBiswas3/PEstats/tree/main/benchmarks). Anyone can reproduce them:

```bash
python benchmarks/rpeaks_mitdb.py      # downloads ~100 MB once
python benchmarks/quality_nstdb.py     # downloads ~45 MB once
```

## R-peak detection — MIT-BIH Arrhythmia Database

All 48 records, channel 0, every annotated beat (109,494), ±150 ms tolerance.

| Detector | Sensitivity | Positive predictivity | F1 |
|---|---|---|---|
| **Pan-Tompkins** (default) | **99.57 %** | **99.11 %** | **99.34 %** |
| Simple threshold | 88.62 % | 93.67 % | 91.08 % |

For reference, Pan & Tompkins (1985) reported 99.75 % / 99.54 %. The hardest records are the well-known ones: 207 (ventricular flutter episodes, which the AAMI EC57 protocol excludes but we keep), 114 (leads swapped, so channel 0 is the low-amplitude V5) and 108 (very large P/T waves and noise).

## Signal quality — MIT-BIH Noise Stress Test Database

NSTDB records `118e*` / `119e*` add real electrode-motion noise to MIT-BIH records 118 and 119 at a known SNR, in alternating 2-minute segments. Each 10-second window therefore has a ground-truth label. False alarms are measured on the untouched originals (360 clean windows).

| SNR | Noisy windows flagged — generic | Noisy windows flagged — `signal_type="ecg"` | AUC (ecg) | Pan-Tompkins F1 |
|---|---|---|---|---|
| 24 dB | 4 % | 5 % | 0.51 | 99.98 % |
| 18 dB | 13 % | 17 % | 0.58 | 99.97 % |
| 12 dB | 42 % | 74 % | 0.89 | 96.94 % |
| 6 dB | 62 % | 91 % | 0.97 | 87.53 % |
| 0 dB | 72 % | 90 % | 0.96 | 80.77 % |
| −6 dB | 74 % | 84 % | 0.93 | 69.11 % |
| **False alarms on clean ECG** | **2.5 %** | **3.6 %** | | |

`quality_timeline(window=10, signal_type="ecg")` on record `118e06` (6 dB) — the flagged blocks line up with the database's noise schedule (noise from 300 s, alternating every 2 minutes):

![Quality timeline on NSTDB record 118e06](img/nstdb_timeline.png)

How to read this:

- Electrode-motion noise sits inside the ECG band, so the generic checks (drift, spikes, mains, …) only catch part of it. The ECG checks — beat plausibility and beat-shape consistency — catch most of it once it matters.
- At 24 and 18 dB beat detection is still essentially perfect, so *not* flagging those windows is the right call. Detection rises exactly where beat detection starts to degrade (≤ 12 dB).
- At −6 dB the noise is beat-like by design and occasionally forms a consistent "rhythm" of its own, which slightly lowers detection.
- Record 119 has ventricular bigeminy (444 PVCs). Irregular rhythm is reported as information, not as a quality failure; an earlier version that treated it as a failure had a 47 % false-alarm rate.

## Readers

The EDF/EDF+/BDF and WFDB readers produce output identical to the reference `pyedflib` and `wfdb` libraries (difference 0.0 on all of MIT-BIH record 100 and on every supported WFDB format; ≤ 3·10⁻¹⁴ on EDF/BDF).

*Data: Moody & Mark, IEEE EMB 20(3) 2001; Moody, Muldrow & Mark, Computers in Cardiology 11, 1984; Goldberger et al., PhysioNet, Circulation 101(23) 2000.*
