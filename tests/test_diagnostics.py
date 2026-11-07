import numpy as np
import pytest

from PBstats import Data, Pipeline, QualityReport

FS = 500


def _codes(x, **kw):
    return {i.code for i in Data(x, fs=FS).diagnose(**kw).quality_report.issues
            if i.severity != "info"}


@pytest.mark.parametrize("fs", [250, 360, 500, 1000])
@pytest.mark.parametrize("noise", [0.0005, 0.01, 0.05])
def test_clean_ecg_has_no_false_alarms(ecg, fs, noise):
    _, x = ecg(fs=fs, noise=noise, seed=fs)
    rep = Data(x, fs=fs).diagnose().quality_report
    assert rep.ok, rep.issues
    assert rep.score >= 90


@pytest.mark.parametrize("make", [
    lambda x, t: np.sin(2 * np.pi * 10 * t),                                   # noiseless sine
    lambda x, t: np.sign(np.sin(2 * np.pi * 2 * t)),                          # square: harmonic at 50 Hz
    lambda x, t: np.random.default_rng(0).standard_normal(len(t)),            # white noise
])
def test_other_signal_families_do_not_alarm(ecg, make):
    t, x = ecg()
    assert _codes(make(x, t)) == set()


def test_detects_each_problem(ecg):
    t, x = ecg()
    rng = np.random.default_rng(1)
    spiky = x.copy()
    pos = rng.choice(len(x), 5, replace=False)
    spiky[pos] += 1.5
    gappy = x.copy()
    gappy[1000:1100] = np.nan
    flat = x.copy()
    flat[3000:3600] = flat[3000]

    assert _codes(x + 0.1 * np.sin(2 * np.pi * 50 * t)) == {"mains"}
    assert _codes(x + 0.1 * np.sin(2 * np.pi * 60 * t)) == {"mains"}
    assert _codes(x + np.sin(2 * np.pi * 0.15 * t)) == {"drift"}
    assert _codes(spiky) == {"spikes"}
    assert _codes(gappy) == {"missing"}
    assert _codes(np.clip(x, None, 0.6)) == {"clipping"}
    assert _codes(flat) == {"flatline"}
    rep = Data(spiky, fs=FS).diagnose().quality_report
    assert set(np.flatnonzero(rep.masks["spikes"])) == set(pos)


def test_mains_frequency_and_fix_are_prescribed(ecg):
    t, x = ecg()
    rep = Data(x + 0.2 * np.sin(2 * np.pi * 60 * t), fs=FS).diagnose().quality_report
    assert rep.metrics["mains_hz"] == 60
    assert ("notch", {"freq": 60.0, "harmonics": False}) in rep.fixes


def test_drift_check_can_be_disabled(ecg):
    t, x = ecg()
    assert "drift" not in _codes(x + np.sin(2 * np.pi * 0.15 * t), drift_cutoff=None)


def test_prescription_order_and_auto_clean(ecg):
    t, clean = ecg()
    rng = np.random.default_rng(2)
    x = clean + 0.2 * np.sin(2 * np.pi * 50 * t) + 0.8 * np.sin(2 * np.pi * 0.15 * t) + 3
    x[rng.choice(len(x), 6, replace=False)] += 5
    x[5000:5050] = np.nan

    d = Data(x, fs=FS).diagnose()
    before = d.quality_report
    assert [f[0] for f in before.fixes] == ["remove_missing", "remove_outliers", "highpass", "notch"]
    assert before.grade in ("D", "F")

    d.auto_clean()
    assert d.quality_report_after.score >= 90
    assert np.corrcoef(d.data, clean)[0, 1] > 0.9
    # R-peak height survives (spike removal must not clip real peaks)
    assert np.ptp(d.data) == pytest.approx(np.ptp(clean), rel=0.15)


def test_prescription_is_a_reusable_pipeline(ecg, tmp_path):
    t, x = ecg()
    noisy = x + 0.2 * np.sin(2 * np.pi * 50 * t)
    pipe = Data(noisy, fs=FS).diagnose().quality_report.pipeline
    assert isinstance(pipe, Pipeline) and len(pipe) == 1
    pipe.save(tmp_path / "rx.json")
    out = Pipeline.load(tmp_path / "rx.json").run(Data(noisy, fs=FS))
    assert Data(out.data, fs=FS).diagnose().quality_report.ok


def test_edge_cases():
    assert Data(np.full(100, np.nan), fs=FS).diagnose().quality_report.score <= 65
    assert Data(np.ones(500), fs=FS).diagnose().quality_report.has("constant")
    assert Data(np.arange(10.0), fs=FS).diagnose().quality_report.has("short")
    # no fs: falls back to trend check
    rep = Data(np.arange(1000.0) + np.random.default_rng(0).standard_normal(1000)).diagnose().quality_report
    assert rep.has("trend") and rep.fixes == [("detrend", {"method": "linear"})]


def test_report_api(ecg, capsys):
    _, x = ecg()
    rep = Data(x, fs=FS, label="ecg").diagnose(verbose=True).quality_report
    assert isinstance(rep, QualityReport)
    assert "Signal diagnosis" in capsys.readouterr().out
    d = rep.to_dict()
    assert d["score"] == rep.score and d["label"] == "ecg"
    assert "score=" in repr(rep)
