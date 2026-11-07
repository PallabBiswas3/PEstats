"""Smoke tests: every plot renders to a file without error, and plotting
does not leak PBstats styling into the user's global matplotlib config."""
import matplotlib as mpl
import numpy as np
import pytest

from PBstats import Data

FS = 250
T = np.arange(4 * FS) / FS
SIG = np.sin(2 * np.pi * 10 * T) + 0.3 * np.random.default_rng(0).standard_normal(len(T))


@pytest.mark.parametrize("make_plot", [
    lambda d, p: d.plot(save_path=p),
    lambda d, p: d.copy().fft().plot_spectrum(save_path=p),
    lambda d, p: d.psd().plot_psd(bands={"alpha": (8, 13)}, save_path=p),
    lambda d, p: d.copy().hilbert().plot_envelope(save_path=p),
    lambda d, p: d.copy().stft(nperseg=64).plot_spectrogram(save_path=p),
    lambda d, p: d.copy().wavelet(n_scales=16, freq_max=60).plot_scalogram(save_path=p),
    lambda d, p: d.describe().plot_stats(save_path=p),
    lambda d, p: d.autocorrelation(max_lag=100).plot_autocorr(save_path=p),
    lambda d, p: d.correlate([SIG * 2, -SIG]).plot_correlation(save_path=p),
    lambda d, p: d.plot_quality(save_path=p),
    lambda d, p: d.plot_filter_response((1, 40), save_path=p),
    lambda d, p: d.plot_filter_response(50, filter_type="notch", save_path=p),
    lambda d, p: d.describe().diagnose().psd().dashboard(save_path=p),
])
def test_plots_render(tmp_path, make_plot):
    path = tmp_path / "fig.png"
    make_plot(Data(SIG, fs=FS, label="test"), str(path))
    assert path.exists() and path.stat().st_size > 1000


def test_multichannel_plot(tmp_path):
    Data(np.c_[SIG, -SIG], fs=FS, channel_names=["a", "b"]).plot(save_path=str(tmp_path / "m.png"))


def test_plotting_does_not_change_global_style(tmp_path):
    before = mpl.rcParams["axes.facecolor"]
    Data(SIG, fs=FS).plot(save_path=str(tmp_path / "s.png"))
    assert mpl.rcParams["axes.facecolor"] == before
