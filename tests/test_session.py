"""Session → Pipeline, Data ergonomics, pipeline hardening, new statistics."""
import json

import numpy as np
import pandas as pd
import pytest

from PBstats import Data, FunctionalPipeline, Pipeline

FS = 250
RNG = np.random.default_rng(7)
X = np.sin(2 * np.pi * 10 * np.arange(5 * FS) / FS) + 0.3 * RNG.standard_normal(5 * FS) + 2


# ── to_pipeline ──────────────────────────────────────────────────────────

def test_to_pipeline_replays_session_exactly(tmp_path):
    d = (Data(X, fs=FS).remove_dc().bandpass(1, 40).notch(50)
         .normalize("zscore").describe())
    pipe = d.to_pipeline("explore")
    # nested calls (normalize → standardize) are not double-recorded
    assert [s.name for s in pipe] == ["remove_dc", "bandpass", "notch", "normalize", "describe"]
    assert pipe.steps[1].kwargs == {"low": 1, "high": 40, "order": 4, "zero_phase": True}

    pipe.save(tmp_path / "p.json")
    replay = Pipeline.load(tmp_path / "p.json").run(Data(X, fs=FS))
    assert np.allclose(replay.data, d.data)
    assert replay.to_pipeline().to_dict()["steps"] == pipe.to_dict()["steps"]


def test_positional_args_are_recorded_by_name():
    d = Data(X, fs=FS).bandpass(2, 30, 6)
    assert d.steps == [("bandpass", {"low": 2, "high": 30, "order": 6, "zero_phase": True})]


# ── Data ergonomics ──────────────────────────────────────────────────────

def test_copy_is_independent():
    a = Data(X, fs=FS).remove_dc()
    b = a.copy().standardize()
    assert len(a.history()) == 1 and len(b.history()) == 2
    assert not np.allclose(a.data, b.data)


def test_channels_and_frame():
    df = pd.DataFrame({"Fz": X, "Cz": -X})
    d = Data(df, fs=FS, label="eeg")
    assert d.n_channels == 2 and d.channel_names == ["Fz", "Cz"]
    cz = d.channel("Cz")
    assert cz.label == "eeg[Cz]" and np.allclose(cz.data, -X)
    assert d.duration == pytest.approx(5.0)
    frame = d.to_frame()
    assert list(frame.columns) == ["Fz", "Cz"] and frame.index.name == "time_s"


def test_single_column_input_is_squeezed():
    assert Data(pd.DataFrame({"a": [1.0, 2.0, 3.0]})).data.ndim == 1


def test_transforms_explain_multichannel_requirement():
    with pytest.raises(ValueError, match="channel"):
        Data(np.c_[X, X], fs=FS).fft()


def test_invalid_inputs():
    with pytest.raises(ValueError):
        Data(X, fs=0)
    with pytest.raises(ValueError):
        Data([])
    with pytest.raises(ValueError):
        Data(np.zeros((2, 2, 2)))


def test_crop_in_seconds():
    assert len(Data(X, fs=FS).crop(1, 3)) == 2 * FS


@pytest.mark.parametrize("method, threshold", [("hampel", 3.0), ("spike", 10.0), ("mad", 3.0)])
def test_outlier_methods_multichannel(method, threshold):
    y = X.copy()
    y[100] = 50
    d = Data(np.c_[y, X], fs=FS).remove_outliers(method=method, threshold=threshold)
    assert d.data[100, 0] < 5            # spike removed, per channel
    assert d.data.shape == (len(X), 2)


def test_spike_method_leaves_clean_data_untouched():
    # Classic Hampel (threshold 3) legitimately trims Gaussian tails;
    # the 'spike' detector must not touch clean data at all.
    y = X.copy()
    y[100] = 50
    d = Data(np.c_[y, X], fs=FS).remove_outliers(method="spike", threshold=10)
    assert np.array_equal(d.data[:, 1], X)
    assert np.array_equal(np.delete(d.data[:, 0], 100), np.delete(y, 100))
    assert "n_removed=1" in d.history()[-1]


# ── Pipeline hardening ──────────────────────────────────────────────────

def test_pipeline_copy_leaves_input_untouched():
    d = Data(X, fs=FS)
    out = Pipeline().add("standardize").run(d, copy=True)
    assert np.allclose(d.data, X) and not np.allclose(out.data, X)


def test_pipeline_refuses_private_methods():
    with pytest.raises(AttributeError):
        Pipeline().add("_record", s="x").run(Data(X))


def test_pipeline_concatenation():
    p = Pipeline("a").add("remove_dc") + Pipeline("b").add("standardize")
    assert p.name == "a+b" and [s.name for s in p] == ["remove_dc", "standardize"]


def test_save_converts_tuples_and_rejects_arrays(tmp_path):
    Pipeline().add("winsorize", limits=(0.01, 0.02)).save(tmp_path / "w.json")
    payload = json.loads((tmp_path / "w.json").read_text())
    assert payload["steps"][0]["kwargs"]["limits"] == [0.01, 0.02]
    assert payload["format_version"] == 1
    with pytest.raises(TypeError, match="cannot be saved"):
        Pipeline().add("ttest", other=np.zeros(3)).save(tmp_path / "bad.json")


def test_functional_pipeline_save_is_explicit_about_callables(tmp_path):
    fp = FunctionalPipeline().add_fn(lambda d: d, name="noop")
    with pytest.raises(ValueError, match="custom callable"):
        fp.save(tmp_path / "f.json")


def test_run_batch_collects_errors():
    pipe = Pipeline().add("bandpass", low=1, high=40)
    out = pipe.run_batch([Data(X, fs=FS), Data(X[:5], fs=FS)], on_error="skip")
    assert len(out) == 1 and pipe.batch_errors[0]["index"] == 1


# ── statistics ───────────────────────────────────────────────────────────

def test_effect_sizes():
    a = RNG.normal(0.8, 1, 400)
    b = RNG.normal(0.0, 1, 400)
    t = Data(a).ttest(b).ttest_result
    assert t.effect_size == pytest.approx(0.8, abs=0.15) and t.reject_null
    mw = Data(a).mannwhitney(b).mannwhitney_result
    assert 0.2 < mw.effect_size < 0.6


def test_cross_correlation_recovers_delay():
    y = RNG.standard_normal(2000)
    r = Data(np.roll(y, 25), fs=FS).cross_correlation(y).xcorr_result
    assert r.best_lag == 25 and r.best_lag_seconds == pytest.approx(25 / FS)


def test_coherence_peaks_at_shared_frequency():
    t = np.arange(20 * FS) / FS
    a = np.sin(2 * np.pi * 12 * t) + RNG.standard_normal(len(t))
    b = np.sin(2 * np.pi * 12 * t + 1) + RNG.standard_normal(len(t))
    c = Data(a, fs=FS).coherence(b).coherence_result
    assert c.peak_frequency() == pytest.approx(12, abs=0.5)
    assert c.band_mean(11, 13) > 3 * c.band_mean(40, 80)


def test_psd_band_power_matches_sine_power():
    t = np.arange(20 * FS) / FS
    p = Data(2 * np.sin(2 * np.pi * 10 * t), fs=FS).psd().psd_result
    assert p.band_power(8, 12) == pytest.approx(2.0, rel=0.05)   # A²/2
    assert p.peak_frequency() == pytest.approx(10, abs=0.5)


def test_detrend_poly_removes_curved_baseline():
    t = np.linspace(0, 1, 1000)
    d = Data(3 * t ** 2 - 2 * t + 0.01 * RNG.standard_normal(1000)).detrend("poly", order=2)
    assert np.std(d.data) < 0.02
