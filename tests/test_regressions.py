"""Regression tests for bugs found in v0.1.x — each one used to fail."""
import time

import numpy as np
import pytest

from PBstats import Data
from PBstats.transforms import FFTResult

FS = 1000
T = np.arange(2 * FS) / FS
SINE = np.sin(2 * np.pi * 50 * T)


def test_peak_and_auto_features_no_longer_crash():
    # used scipy.stats.find_peaks, which does not exist
    d = Data(SINE, fs=FS).auto_features()
    assert d.feature_vector.features["pk_count"] > 0
    assert len(d.feature_vector) > 30


@pytest.mark.parametrize("window", ["hann", "hamming", "blackman", "boxcar"])
def test_fft_amplitude_is_calibrated(window):
    # was 0.5 for a unit sine with a Hann window (coherent gain ignored)
    mag = Data(2.5 * SINE, fs=FS).fft(window=window).fft_result.magnitude
    assert mag.max() == pytest.approx(2.5, rel=1e-3)


def test_impute_median_on_1d():
    assert np.allclose(Data([1, np.nan, 3]).impute("median").data, [1, 2, 3])


def test_filters_run_along_time_on_multichannel():
    x = np.c_[SINE, SINE + 5]
    d = Data(x, fs=FS).highpass(1)
    # channel 2's +5 offset removed, channel shapes preserved
    assert d.data.shape == x.shape
    assert abs(np.median(d.data[:, 1])) < 0.1
    assert np.allclose(d.data[:, 0], d.data[:, 1], atol=0.05)


def test_remove_missing_on_multichannel():
    d = Data(np.array([[1, np.nan], [2, 3.0], [3, 4]])).remove_missing()
    assert not np.isnan(d.data).any()
    assert d.data[0, 1] == 3.0


def test_reset_restores_fs_and_clears_results():
    d = Data(SINE, fs=FS).resample(500).fft()
    d.reset()
    assert d.fs == FS
    assert len(d.data) == len(SINE)
    assert not hasattr(d, "fft_result")
    assert d.history() == [] and d.steps == []


def test_resample_non_integer_rate():
    d = Data(np.zeros(1002), fs=250.5).resample(125)
    assert len(d.data) == pytest.approx(1002 * 125 / 250.5, abs=1)
    assert d.fs == 125


def test_result_classes_are_unified():
    assert isinstance(Data(SINE, fs=FS).fft().fft_result, FFTResult)


def test_sample_entropy_is_fast():
    # was ~16 s for 1000 samples (O(n²) Python loop)
    x = np.random.default_rng(0).standard_normal(5000)
    t0 = time.perf_counter()
    Data(x, fs=FS).nonlinear_features()
    assert time.perf_counter() - t0 < 5


def test_bandstop_validates_band():
    with pytest.raises(ValueError):
        Data(SINE, fs=FS).bandstop(60, 40)
    with pytest.raises(ValueError):
        Data(SINE, fs=FS).bandstop(400, 600)   # above Nyquist


def test_short_signal_gives_clear_filter_error():
    with pytest.raises(ValueError, match="needs at least"):
        Data(np.ones(10), fs=FS).bandpass(10, 50)


def test_plot_envelope_uses_transformed_signal_not_raw():
    d = Data(SINE + 3, fs=FS).remove_dc().hilbert()
    assert np.allclose(d.hilbert_result.signal, SINE + 3 - np.mean(SINE + 3))


def test_autocorrelation_matches_definition():
    x = np.random.default_rng(1).standard_normal(500)
    ac = Data(x).autocorrelation(max_lag=5).autocorr
    xc = x - x.mean()
    expected = [np.sum(xc[: len(x) - k] * xc[k:]) / np.sum(xc * xc) for k in range(6)]
    assert np.allclose(ac, expected)


def test_pnn50_and_rr_peaks_use_correct_definitions(ecg):
    _, x = ecg(fs=500, dur=30)
    f = Data(x, fs=500).hrv_features().feature_vector.features
    assert 60 < f["hrv_mean_hr_bpm"] < 80
    assert 0 <= f["hrv_pnn50_pct"] <= 100
