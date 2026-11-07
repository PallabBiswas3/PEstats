# PBstats

**From a raw, messy recording to ML-ready features — with a diagnosis, a reproducible recipe, and a model at the end.**

PBstats is a signal-aware analysis library for time series and biosignals (ECG, EEG, EMG, vibration, sensors, spectroscopy). One chainable `Data` object takes you through loading, diagnosis, cleaning, transforms, statistics, feature extraction and machine learning.

```python
from PBstats import Data

d = Data.from_physionet("mitdb/100", channels="MLII")   # download + read a real ECG
d.diagnose(signal_type="ecg", verbose=True)             # what's wrong with it?
d.auto_clean()                                          # apply the prescribed fixes
d.report("mitdb_100.html", signal_type="ecg")           # one-file shareable report
```

## What makes it different

<div class="grid" markdown>

**It tells you what's wrong.**
`diagnose()` checks for gaps, dropout, clipping, spikes, drift, mains interference, noise and — for ECG — implausible rhythms and inconsistent beat shapes. It grades the recording 0–100 and *prescribes* a cleaning pipeline. [Validated on PhysioNet](benchmarks.md).

**Your exploration becomes your pipeline.**
Every chained call is recorded, so `d.to_pipeline()` turns what you tried interactively into a saveable, replayable pipeline for a whole dataset.

**It plugs into machine learning.**
`SignalFeatureExtractor` is a scikit-learn transformer; `Augmenter` and `make_windows` prepare deep-learning data with realistic artefacts.

**It reads real data.**
CSV, WAV, EDF/EDF+/BDF and PhysioNet WFDB — pure Python, no extra dependencies, verified bit-for-bit against the reference libraries.

</div>

## Install

```bash
pip install PBstats
```

Optional extras: `pip install "PBstats[dl]"` (PyTorch datasets), `"PBstats[docs]"`.

Next: [Getting started](getting-started.md).
