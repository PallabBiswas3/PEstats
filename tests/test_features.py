import numpy as np
import pandas as pd
import pytest

from PBstats import Data
from PBstats.features import (
    dfa_alpha,
    extract,
    higuchi_fd,
    permutation_entropy,
    sample_entropy,
)

RNG = np.random.default_rng(42)
WHITE = RNG.standard_normal(5000)
BROWN = np.cumsum(RNG.standard_normal(5000))
SINE = np.sin(2 * np.pi * 5 * np.arange(5000) / 500)


def _sampen_naive(x, m=2):
    n, r = len(x), 0.2 * np.std(x)

    def count(length):
        T = np.array([x[i:i + length] for i in range(n - m)])
        return sum((np.max(np.abs(T - T[i]), axis=1) <= r).sum() - 1 for i in range(len(T)))
    return -np.log(count(m + 1) / count(m))


def test_sample_entropy_matches_reference():
    x = WHITE[:400]
    assert sample_entropy(x) == pytest.approx(_sampen_naive(x), rel=1e-12)


def test_sample_entropy_orders_regular_below_random():
    assert sample_entropy(SINE) < sample_entropy(WHITE)


def test_sample_entropy_undefined_is_nan():
    assert np.isnan(sample_entropy(np.ones(100)))


def test_permutation_entropy_bounds():
    assert permutation_entropy(WHITE) == pytest.approx(1.0, abs=0.01)
    assert permutation_entropy(np.arange(100.0)) == 0.0


@pytest.mark.parametrize("x, expected", [(SINE, 1.0), (BROWN, 1.5), (WHITE, 2.0)])
def test_higuchi_fd_known_values(x, expected):
    assert higuchi_fd(x) == pytest.approx(expected, abs=0.1)


@pytest.mark.parametrize("x, expected", [(WHITE, 0.5), (BROWN, 1.5)])
def test_dfa_known_values(x, expected):
    assert dfa_alpha(x) == pytest.approx(expected, abs=0.15)


def test_frequency_features_find_dominant_band():
    fs = 256
    t = np.arange(30 * fs) / fs
    alpha = np.sin(2 * np.pi * 10 * t) + 0.3 * np.random.default_rng(0).standard_normal(len(t))
    f = extract(alpha, fs, include=["freq"])
    assert f["fd_dominant_frequency"] == pytest.approx(10, abs=0.5)
    assert f["fd_band_alpha"] > 0.5
    assert f["fd_band_alpha"] > max(f["fd_band_delta"], f["fd_band_beta"])


def test_feature_names_are_stable_across_signals():
    a = extract(WHITE[:1000], 100)
    b = extract(SINE[:3000], 100)
    assert list(a) == list(b)


def test_extract_rejects_nan_and_unknown_category():
    with pytest.raises(ValueError, match="NaN"):
        extract(np.array([1.0, np.nan, 2.0] * 20), 100)
    with pytest.raises(ValueError, match="Unknown"):
        extract(WHITE, 100, include=["bogus"])


def test_hrv_features(ecg):
    _, x = ecg(fs=500, dur=120, hr=60)
    f = Data(x, fs=500).bandpass(0.5, 40).hrv_features().feature_vector.features
    assert f["hrv_mean_hr_bpm"] == pytest.approx(60, abs=3)
    assert f["hrv_beats"] == pytest.approx(120, abs=4)
    for key in ("hrv_sdnn_ms", "hrv_rmssd_ms", "hrv_sd1_ms", "hrv_sd2_ms", "hrv_lf_hf_ratio"):
        assert np.isfinite(f[key]) and f[key] > 0


def test_hrv_too_few_beats_warns_and_keeps_schema():
    with pytest.warns(UserWarning, match="R-peaks"):
        f = Data(np.zeros(1000), fs=500).hrv_features().feature_vector.features
    assert np.isnan(f["hrv_sdnn_ms"]) and "hrv_lf_hf_ratio" in f


def test_multichannel_features_are_prefixed():
    df = pd.DataFrame({"Fz": WHITE[:2000], "Cz": SINE[:2000]})
    f = Data(df, fs=500).time_features().feature_vector.features
    assert "Fz__td_rms" in f and "Cz__td_rms" in f
    assert f["Cz__td_rms"] == pytest.approx(np.sqrt(0.5), rel=0.01)


def test_windowed_features():
    fs = 100
    x = np.r_[np.sin(2 * np.pi * 5 * np.arange(10 * fs) / fs),
              np.sin(2 * np.pi * 20 * np.arange(10 * fs) / fs)]
    df = Data(x, fs=fs, label="rec").windowed_features(window=2, step=1, include=["freq"])
    assert len(df) == 19
    assert list(df.columns[:3]) == ["label", "t_start", "t_end"]
    assert df["fd_dominant_frequency"].iloc[0] == pytest.approx(5, abs=0.5)
    assert df["fd_dominant_frequency"].iloc[-1] == pytest.approx(20, abs=0.5)


def test_hrv_rejects_artefactual_rr_intervals():
    rng = np.random.default_rng(0)
    rr = 850 + 30 * rng.standard_normal(60)
    bad = rr.copy()
    bad[20:22] = [314, 536]                       # an extra (false) beat splits one interval
    from PBstats.features.functions import hrv
    clean, filtered, raw = hrv(rr), hrv(bad), hrv(bad, rr_filter=None)
    assert filtered["hrv_sdnn_ms"] == pytest.approx(clean["hrv_sdnn_ms"], rel=0.1)
    assert raw["hrv_sdnn_ms"] > 2 * clean["hrv_sdnn_ms"]
    assert filtered["hrv_artifact_pct"] == pytest.approx(100 * 2 / 60, abs=0.5)


def test_hrv_rejects_artefact_at_record_edge():
    from PBstats.features.functions import clean_rr
    rr = np.r_[850 + 30 * np.random.default_rng(1).standard_normal(30), 314]
    assert not clean_rr(rr)[-1] and clean_rr(rr)[:-1].all()
