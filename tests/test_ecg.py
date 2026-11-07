import numpy as np
import pytest
from conftest import synth_ecg

from PBstats import Data
from PBstats.features.ecg import detect_r_peaks, match_beats, pan_tompkins


def _true_r(fs, dur, hr, seed):
    """R-peak positions of synth_ecg (the R wave is the tallest Gaussian)."""
    t, x = synth_ecg(fs=fs, dur=dur, hr=hr, seed=seed, noise=0.0)
    from scipy.signal import find_peaks
    return find_peaks(x, height=0.6)[0]


@pytest.mark.parametrize("fs", [250, 360, 500, 1000])
def test_pan_tompkins_finds_every_beat(fs):
    _, x = synth_ecg(fs=fs, dur=60, seed=fs, noise=0.03)
    s = match_beats(_true_r(fs, 60, 70, fs), pan_tompkins(x, fs), fs)
    assert s["Se"] == 1.0 and s["PPV"] == 1.0


def test_pan_tompkins_handles_inverted_polarity_and_drift():
    fs = 360
    t, x = synth_ecg(fs=fs, dur=60, seed=1, noise=0.03)
    bad = -x + 0.8 * np.sin(2 * np.pi * 0.2 * t)
    s = match_beats(_true_r(fs, 60, 70, 1), pan_tompkins(bad, fs), fs)
    assert s["F1"] > 0.99
    # the simple threshold detector cannot cope with this
    assert match_beats(_true_r(fs, 60, 70, 1), detect_r_peaks(bad, fs, method="simple"), fs)["F1"] < 0.9


def test_match_beats_scoring():
    s = match_beats([100, 400, 700], [102, 390, 1000], fs=100, tolerance=0.15)  # ±15 samples
    assert (s["TP"], s["FN"], s["FP"]) == (2, 1, 1)
    assert match_beats([100], [], fs=100)["F1"] == 0.0


def test_hrv_features_use_pan_tompkins_by_default():
    fs = 360
    t, x = synth_ecg(fs=fs, dur=120, hr=60, seed=2)
    d = Data(-x, fs=fs).hrv_features()                 # inverted lead
    assert d.feature_vector.features["hrv_mean_hr_bpm"] == pytest.approx(60, abs=3)


# ── ECG signal-quality checks ────────────────────────────────────────────

def _ecg_codes(x, fs):
    rep = Data(x, fs=fs).diagnose(signal_type="ecg").quality_report
    return {i.code for i in rep.issues if i.severity != "info"}, rep


def test_clean_ecg_passes_ecg_checks():
    _, x = synth_ecg(fs=360, dur=30, seed=4)
    codes, rep = _ecg_codes(x, 360)
    assert codes == set()
    assert rep.metrics["ecg_hr_bpm"] == pytest.approx(70, abs=3)
    assert rep.metrics["ecg_template_corr"] > 0.95


def test_inband_noise_is_flagged_by_morphology_check():
    fs = 360
    _, x = synth_ecg(fs=fs, dur=30, seed=5)
    rng = np.random.default_rng(0)
    from scipy.signal import butter, sosfiltfilt
    emg = sosfiltfilt(butter(4, [5, 40], btype="bandpass", fs=fs, output="sos"), rng.standard_normal(len(x)))
    noisy = x + 0.6 * emg / emg.std()
    codes, _ = _ecg_codes(noisy, fs)
    assert "ecg_morphology" in codes or "ecg_rhythm" in codes
    assert _ecg_codes(noisy, fs)[1].score < _ecg_codes(x, fs)[1].score


def test_bigeminy_is_not_a_quality_failure():
    """Every other beat ectopic (different shape, early) — clinical content,
    not noise: reported as 'irregular' info, not as a warning."""
    fs = 360
    t = np.arange(30 * fs) / fs
    x = np.zeros_like(t)
    beat, k = 0.5, 0
    while beat < 29.5:
        if k % 2 == 0:
            x += np.exp(-((t - beat) ** 2) / (2 * 0.01 ** 2))
            beat += 0.45
        else:   # wide, tall ventricular beat after a short interval
            x += 1.8 * np.exp(-((t - beat) ** 2) / (2 * 0.03 ** 2))
            beat += 1.1
        k += 1
    codes, rep = _ecg_codes(x + 0.01 * np.random.default_rng(1).standard_normal(len(x)), fs)
    assert codes == set()
    assert rep.has("ecg_irregular")


def test_quality_timeline_localises_noise():
    fs = 360
    _, x = synth_ecg(fs=fs, dur=60, seed=6)
    x = x.copy()
    x[20 * fs:30 * fs] += 0.8 * np.random.default_rng(2).standard_normal(10 * fs)
    tl = Data(x, fs=fs).quality_timeline(window=10, signal_type="ecg")
    assert len(tl) == 6
    assert list(tl["ok"]) == [True, True, False, True, True, True]
