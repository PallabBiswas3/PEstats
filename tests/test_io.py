import numpy as np
import pandas as pd
import pytest
from fileformats import write_edf, write_wfdb

from PBstats import Data
from PBstats.io import read_annotations

RNG = np.random.default_rng(3)


# ── CSV ──────────────────────────────────────────────────────────────────

def test_csv_infers_fs_from_numeric_time(tmp_path):
    t = np.arange(1000) / 250
    pd.DataFrame({"t": t, "a": np.sin(t), "b": np.cos(t)}).to_csv(tmp_path / "x.csv", index=False)
    d = Data.from_csv(tmp_path / "x.csv", time_col="t")
    assert d.fs == pytest.approx(250)
    assert d.channel_names == ["a", "b"] and d.label == "x"


def test_csv_datetime_time_and_column_selection(tmp_path):
    ts = pd.date_range("2026-01-01", periods=600, freq="100ms")
    pd.DataFrame({"time": ts, "temp": RNG.normal(size=600), "id": "s1"}).to_csv(tmp_path / "s.csv", index=False)
    d = Data.from_csv(tmp_path / "s.csv", time_col="time", columns=["temp"])
    assert d.fs == pytest.approx(10) and d.data.ndim == 1
    assert d.meta["start_time"] == ts[0]


def test_csv_irregular_sampling_warns_or_regularizes(tmp_path):
    t = np.cumsum(RNG.uniform(0.008, 0.012, 500))
    pd.DataFrame({"t": t, "v": np.sin(t)}).to_csv(tmp_path / "j.csv", index=False)
    with pytest.warns(UserWarning, match="Irregular"):
        Data.from_csv(tmp_path / "j.csv", time_col="t")
    d = Data.from_csv(tmp_path / "j.csv", time_col="t", regularize=True)
    assert np.allclose(d.data, np.sin(t[0] + np.arange(len(d.data)) / d.fs), atol=1e-3)


# ── WAV ──────────────────────────────────────────────────────────────────

def test_wav_roundtrip(tmp_path):
    from scipy.io import wavfile
    x = (0.5 * np.sin(2 * np.pi * 440 * np.arange(8000) / 8000) * 32767).astype(np.int16)
    wavfile.write(tmp_path / "a.wav", 8000, np.c_[x, x // 2])
    d = Data.from_wav(tmp_path / "a.wav")
    assert d.fs == 8000 and d.channel_names == ["left", "right"]
    assert np.max(np.abs(d.data[:, 0])) == pytest.approx(0.5, abs=1e-3)


# ── EDF+ ─────────────────────────────────────────────────────────────────

def test_edf_signals_and_annotations(tmp_path):
    fs, n = 128, 128 * 20
    sig = {"Fz": 40 * np.sin(2 * np.pi * 10 * np.arange(n) / fs), "Cz": RNG.normal(0, 15, n)}
    pmin, pmax = write_edf(tmp_path / "e.edf", sig, fs,
                           annotations=[(2.5, 1.0, "stim A"), (11.0, 0, "eyes closed")])
    d = Data.from_edf(tmp_path / "e.edf")
    assert d.fs == fs and d.channel_names == ["Fz", "Cz"] and d.data.shape == (n, 2)
    quantum = max((hi - lo) / 65535 for lo, hi in zip(pmin, pmax))
    assert np.max(np.abs(d.data - np.c_[sig["Fz"], sig["Cz"]])) <= quantum
    assert d.annotations[["onset", "label"]].values.tolist() == [[2.5, "stim A"], [11.0, "eyes closed"]]
    assert Data.from_edf(tmp_path / "e.edf", channels="Cz").data.ndim == 1


# ── WFDB ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fmt", [212, 16])
def test_wfdb_signal_and_annotations(tmp_path, fmt):
    fs = 360
    sig = np.c_[np.sin(np.arange(3600) / 30), 0.5 * np.cos(np.arange(3600) / 50)]
    beats = [(100, "N"), (400, "N"), (700, "V"), (2900, "N"), (3500, "+")]
    dig = write_wfdb(tmp_path / "r", sig, fs, fmt=fmt, annotations=beats)
    d = Data.from_wfdb(tmp_path / "r")
    assert d.fs == fs and d.channel_names == ["MLII", "V5"]
    assert np.array_equal(np.round(d.data * 200).astype(int), dig)
    ann = read_annotations(tmp_path / "r")
    assert list(ann["sample"]) == [b[0] for b in beats]          # incl. a SKIP-coded gap
    assert list(ann["label"]) == [b[1] for b in beats]

    w = Data.from_wfdb(tmp_path / "r", channels="V5", start=1, end=5)
    assert len(w.data) == 4 * fs
    # window 1–5 s = samples 360..1800 → beats at 400 and 700, re-timed to the window
    assert w.annotations["onset"].tolist() == pytest.approx([(400 - 360) / fs, (700 - 360) / fs])
    assert w.annotations["label"].tolist() == ["N", "V"]


def test_crop_shifts_annotations_and_reset_restores(tmp_path):
    ann = pd.DataFrame({"onset": [1.0, 5.0, 9.0], "label": ["a", "b", "c"]})
    d = Data(np.zeros(1000), fs=100, annotations=ann).crop(4, 8)
    assert d.annotations["onset"].tolist() == [1.0] and d.annotations["label"].tolist() == ["b"]
    d.reset()
    assert d.annotations["onset"].tolist() == [1.0, 5.0, 9.0]
    with pytest.raises(ValueError, match="onset"):
        Data(np.zeros(10), annotations=pd.DataFrame({"t": [1]}))
