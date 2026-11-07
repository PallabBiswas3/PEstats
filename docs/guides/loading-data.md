# Loading data

All readers are pure Python/NumPy (no extra packages) and return a `Data` object. Events stored in the file are attached as `d.annotations` — a DataFrame with `onset` (seconds), `duration`, `label` and `aux` — and stay aligned through `crop()`.

| Format | Call | Notes |
|---|---|---|
| CSV | `Data.from_csv(path, time_col="t")` | `fs` inferred from timestamps (numeric or datetime). Irregular sampling warns; `regularize=True` interpolates onto a uniform grid. |
| WAV | `Data.from_wav(path)` | Integer PCM scaled to [-1, 1). Stereo → channels `left`, `right`. |
| EDF / EDF+ / BDF | `Data.from_edf(path, channels=[...])` | EEG/PSG standard. EDF+ annotations become `d.annotations`. Channels must share a sampling rate; by default the most common rate is loaded. |
| WFDB (PhysioNet) | `Data.from_wfdb("data/mitdb/100")` | Formats 16, 24, 32, 61, 80, 212. Beat/rhythm labels from `.atr` files. `start=` / `end=` in seconds. |
| PhysioNet download | `Data.from_physionet("mitdb/100")` | Downloads once into `~/.cache/pbstats` (or `$PBSTATS_DATA`), then reads locally. |

The EDF and WFDB readers were verified to give identical output to the reference `pyedflib` and `wfdb` libraries (all 650,000 samples and 2,274 annotations of MIT-BIH record 100; every supported WFDB format).

## Examples

```python
from PBstats import Data

ecg = Data.from_physionet("mitdb/119", channels="MLII")
ecg.annotations.label.value_counts()        # N 1543, V 444, ...

eeg = Data.from_edf("sleep.edf", channels=["EEG Fpz-Cz", "EEG Pz-Oz"])
eeg.annotations                              # sleep stages / events from EDF+

log = Data.from_csv("sensor_log.csv", time_col="timestamp", columns=["temp", "vibration"])
```

!!! note "Data licences"
    PhysioNet databases carry their own licences (most are ODC-By). `download_physionet` only fetches files; using them is subject to those terms.

Working with your own arrays is always possible: `Data(array, fs=..., annotations=df)`.
