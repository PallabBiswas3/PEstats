"""Change points, anomalies, epochs, forecasting, augmentation, reports."""
import numpy as np
import pandas as pd
import pytest

from PBstats import Augmenter, Data, Epochs, make_windows
from PBstats.augment import add_noise, add_spikes, dropout, time_stretch
from PBstats.events import pelt



# ── change points ────────────────────────────────────────────────────────

def _exhaustive(x, pen, ms):
    n = len(x)
    F = np.full(n + 1, np.inf)
    F[0] = -pen
    last = np.zeros(n + 1, int)
    for t in range(ms, n + 1):
        for s in range(0, t - ms + 1):
            if np.isfinite(F[s]):
                v = max(x[s:t].var(), 1e-6 * x.var())
                c = F[s] + (t - s) * (np.log(2 * np.pi * v) + 1) + pen
                if c < F[t]:
                    F[t], last[t] = c, s
    cps, t = [], n
    while t > 0:
        t = last[t]
        if t > 0:
            cps.append(t)
    return sorted(cps)


def test_pelt_is_exact():
    rng = np.random.default_rng(1)
    z = np.r_[rng.normal(0, 1, 60), rng.normal(2, 0.5, 50), rng.normal(0, 2, 70)]
    cps, pen = pelt(z, "meanvar", min_size=5)
    assert list(cps) == _exhaustive(z, pen, 5)


def test_change_points_recovered_and_none_in_noise():
    rng = np.random.default_rng(2)
    x = np.r_[rng.normal(0, 1, 3000), rng.normal(0, 3, 3000), rng.normal(2, 1, 3000)]
    r = Data(x, fs=100).detect_change_points().change_point_result
    assert np.allclose(r.times, [30, 60], atol=0.2)
    assert len(r.segments) == 3 and r.segments["std"].iloc[1] == pytest.approx(3, rel=0.1)
    assert len(Data(rng.normal(size=5000), fs=100).detect_change_points().change_point_result) == 0


def test_change_points_on_long_signal_use_block_grid():
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(0, 1, 200_000), rng.normal(1, 1, 100_000)]
    r = Data(x, fs=1000).detect_change_points(cost="mean").change_point_result
    assert len(r) == 1 and abs(r.indices[0] - 200_000) <= 300


def test_anomaly_scores_find_the_odd_window():
    rng = np.random.default_rng(4)
    x = rng.normal(size=12_000)
    x[6000:6100] += 12 * np.sin(np.arange(100))
    res = Data(x, fs=100).anomaly_scores(window=1)
    assert res.sort_values("score").iloc[-1]["t_start"] == 60.0
    assert res["anomaly"].dtype == bool


# ── epochs ───────────────────────────────────────────────────────────────

def test_epochs_average_recovers_evoked_response():
    rng = np.random.default_rng(5)
    fs = 250
    n = 120 * fs
    x = rng.normal(0, 2, n)                                     # noise 2x the response
    onsets = np.arange(2, 118, 1.0)
    erp = np.exp(-((np.arange(-50, 200) / fs - 0.3) ** 2) / (2 * 0.08 ** 2))
    for o in onsets:
        c = int(o * fs)
        x[c - 50:c + 200] += erp
    ann = pd.DataFrame({"onset": onsets, "label": ["stim"] * len(onsets)})
    d = Data(x, fs=fs, annotations=ann)
    ep = d.epochs("stim", tmin=-0.2, tmax=0.8)
    assert isinstance(ep, Epochs) and len(ep) == len(onsets)
    avg = ep.average()
    # epochs span -0.2..+0.8 s inclusive (251 samples); the response covers 250.
    # One epoch barely shows the response; averaging 116 cuts noise by ~sqrt(116).
    single = np.corrcoef(ep.data[0][:250], erp)[0, 1]
    averaged = np.corrcoef(avg.data[:250], erp)[0, 1]
    assert averaged > 0.8 and averaged > 3 * single
    assert ep.times[np.argmax(avg.data)] == pytest.approx(0.3, abs=0.06)
    assert ep.to_numpy().shape == (len(onsets), 1, len(ep.times))


def test_epochs_drop_edges_select_labels_and_reject():
    ann = pd.DataFrame({"onset": [0.05, 2.0, 4.0, 9.98], "label": ["N", "V", "N", "N"]})
    x = np.zeros(1000)
    x[395:405] = 100                                           # artefact in the 3rd epoch
    d = Data(x, fs=100, annotations=ann)
    ep = d.epochs(tmin=-0.2, tmax=0.5)
    assert len(ep) == 2 and ep.n_dropped == 2
    assert len(ep["V"]) == 1
    assert len(d.epochs(tmin=-0.2, tmax=0.5, reject=50)) == 1
    assert len(d.epochs([2.0, 4.0], tmin=-0.1, tmax=0.1)) == 2


# ── forecasting ──────────────────────────────────────────────────────────

def test_forecast_learns_trend_and_cycle():
    rng = np.random.default_rng(5)
    t = np.arange(600) / 10
    y = np.sin(2 * np.pi * 0.2 * t) + 0.1 * t + 0.05 * rng.standard_normal(600)
    f = Data(y, fs=10).forecast(holdout=30).forecast_result
    m = f.metrics
    assert m["skill"] > 0.5 and m["coverage"] >= 0.8
    assert len(f.mean) == 30 and f.times[0] == pytest.approx(57.0)


@pytest.mark.parametrize("seed", [0, 6])
def test_forecast_never_diverges(seed):
    """Seed 6 used to pick a model that diverged after refitting."""
    rng = np.random.default_rng(seed)
    t = np.arange(600) / 10
    y = np.sin(2 * np.pi * 0.2 * t) + 0.05 * t + 0.1 * rng.standard_normal(600)
    m = Data(y, fs=10).forecast(holdout=50, n_jobs=2).forecast_result.metrics
    assert m["skill"] > -0.25 and np.isfinite(m["MAE"])


def test_forecast_with_fixed_order_and_horizon():
    rng = np.random.default_rng(7)
    y = np.cumsum(rng.normal(size=300))
    f = Data(y, fs=2).forecast(horizon=5, order=(1, 1, 0)).forecast_result
    assert len(f.mean) == 10 and f.order == (1, 1, 0) and f.metrics == {}
    assert np.all(f.lower <= f.mean) and np.all(f.mean <= f.upper)


# ── augmentation ─────────────────────────────────────────────────────────

def test_noise_snr_is_accurate():
    x = np.sin(np.linspace(0, 200, 20000))
    y = add_noise(x, snr_db=10, rng=0)
    snr = 10 * np.log10(np.mean(x ** 2) / np.mean((y - x) ** 2))
    assert snr == pytest.approx(10, abs=0.3)


def test_transforms_shapes_and_effects():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(2, 1000))
    assert add_spikes(x, n=3, rng=0).shape == x.shape
    assert (dropout(x, 0.5, rng=1) == 0).any()
    assert time_stretch(x, 1.2).shape == x.shape


def test_augmenter_is_reproducible_and_expands():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(5, 1, 500))
    a1 = Augmenter(fs=100, powerline=1, wander=1, seed=3).expand(X, copies=2)
    a2 = Augmenter(fs=100, powerline=1, wander=1, seed=3).expand(X, copies=2)
    assert a1.shape == (15, 1, 500) and a1.dtype == np.float32 and np.array_equal(a1, a2)
    with pytest.raises(ValueError, match="fs"):
        Augmenter(powerline=0.5)


def test_augmented_artefacts_are_what_diagnose_detects():
    from conftest import synth_ecg
    fs = 360
    _, x = synth_ecg(fs=fs, dur=20)
    aug = Augmenter(fs=fs, powerline=1, wander=1, spikes=1, noise_snr=None, scale_range=None, seed=1)
    codes = {i.code for i in Data(aug(x), fs=fs).diagnose().quality_report.issues}
    assert {"mains", "drift", "spikes"} <= codes


def test_make_windows_and_torch_dataset():
    rng = np.random.default_rng(10)
    a = Data(rng.normal(size=(1000, 2)), fs=100)
    b = rng.normal(size=700)
    X, y = make_windows([a, Data(rng.normal(size=(700, 2)), fs=100)], window=2, step=1,
                        labels=["a", "b"], layout="NTC")
    assert X.shape == (9 + 6, 200, 2) and X.dtype == np.float32
    assert list(y).count("a") == 9 and list(y).count("b") == 6
    with pytest.raises(ValueError, match="channels"):
        make_windows([a, Data(b, fs=100)], window=2)
    X1, y1 = make_windows(Data(b, fs=100), window=2, labels="b")
    assert X1.shape == (3, 1, 200) and list(y1) == ["b"] * 3
    torch = pytest.importorskip("torch")
    from PBstats import to_torch_dataset
    ds = to_torch_dataset(X1, np.array([0, 1, 0]), augmenter=Augmenter(seed=0))
    xb, yb = ds[1]
    assert isinstance(xb, torch.Tensor) and tuple(xb.shape) == (1, 200) and int(yb) == 1


# ── report ───────────────────────────────────────────────────────────────

def test_report_is_self_contained(tmp_path):
    from conftest import synth_ecg
    fs = 250
    t, x = synth_ecg(fs=fs, dur=70)
    d = Data(x + 0.2 * np.sin(2 * np.pi * 50 * t), fs=fs, label="pt <1>").bandpass(0.5, 100)
    html = d.report(tmp_path / "r.html", signal_type="ecg")
    text = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert text == html
    assert "pt &lt;1&gt;" in html                                    # escaped
    assert html.count("data:image/png;base64,") >= 3                 # embedded figures
    assert "Quality over time" in html and "hrv_rmssd_ms" in html
    assert "bandpass" in html                                        # processing history
    assert "http://" not in html and "https://" not in html          # no external assets
