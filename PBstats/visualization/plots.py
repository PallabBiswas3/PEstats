from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from PBstats.visualization.base import COLORS, save_or_show, styled


@styled
def plot_signal(
    data: np.ndarray,
    fs: float = 1.0,
    label: str = "signal",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
    channel_names: list[str] | None = None,
) -> plt.Figure:
    """Time-domain plot. 2D data (n_samples, n_channels) draws one line per
    channel. If ``ax`` is given, draws into it (for subplots)."""
    standalone = ax is None
    fig, ax = plt.subplots(figsize=(12, 3)) if standalone else (ax.get_figure(), ax)

    data = np.asarray(data)
    t = np.arange(data.shape[0]) / fs
    if data.ndim == 1:
        ax.plot(t, data, color=COLORS["primary"], lw=1.2, alpha=0.9)
    else:
        names = channel_names or [f"ch{i}" for i in range(data.shape[1])]
        for i in range(data.shape[1]):
            ax.plot(t, data[:, i], lw=1.0, alpha=0.85, label=names[i])
        ax.legend(fontsize=8, frameon=False, ncol=min(data.shape[1], 6))
    ax.set_xlabel("Time (s)" if fs != 1.0 else "Sample")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{label} — time domain")
    if len(t) > 1:
        ax.set_xlim(t[0], t[-1])

    if standalone:
        save_or_show(fig, save_path)
    return fig


@styled
def plot_fft(
    fft_result,
    max_freq: float | None = None,
    log_scale: bool = False,
    mark_peaks: int = 3,
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """FFT amplitude spectrum with the N strongest *peaks* annotated
    (local maxima, not adjacent bins of the same peak)."""
    from scipy.signal import find_peaks

    standalone = ax is None
    fig, ax = plt.subplots(figsize=(12, 4)) if standalone else (ax.get_figure(), ax)

    freqs, mag = fft_result.freqs, fft_result.magnitude
    if max_freq:
        mask = freqs <= max_freq
        freqs, mag = freqs[mask], mag[mask]

    ax.fill_between(freqs, mag, alpha=0.15, color=COLORS["primary"])
    ax.plot(freqs, mag, color=COLORS["primary"], lw=1.3)
    if log_scale:
        ax.set_yscale("log")

    if mark_peaks > 0 and len(mag) > 2:
        pk, _ = find_peaks(mag)
        for idx in pk[np.argsort(mag[pk])[::-1][:mark_peaks]]:
            ax.axvline(freqs[idx], color=COLORS["accent"], lw=1.0, ls="--", alpha=0.7)
            ax.annotate(f"{freqs[idx]:.1f} Hz", xy=(freqs[idx], mag[idx]),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=9, color=COLORS["accent"])

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{fft_result.label} — frequency spectrum")
    ax.set_xlim(0, freqs[-1])

    if standalone:
        save_or_show(fig, save_path)
    return fig


@styled
def plot_psd(
    psd_result,
    max_freq: float | None = None,
    bands: dict | None = None,
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Welch PSD in dB, optionally shading frequency bands
    (e.g. ``{"alpha": (8, 13)}``)."""
    standalone = ax is None
    fig, ax = plt.subplots(figsize=(12, 4)) if standalone else (ax.get_figure(), ax)

    f, db = psd_result.freqs, psd_result.to_db()
    if max_freq:
        mask = f <= max_freq
        f, db = f[mask], db[mask]
    ax.plot(f, db, color=COLORS["primary"], lw=1.4)
    if bands:
        palette = [COLORS["secondary"], COLORS["accent"], COLORS["warning"], COLORS["neutral"]]
        for i, (name, (lo, hi)) in enumerate(bands.items()):
            ax.axvspan(lo, hi, color=palette[i % len(palette)], alpha=0.12)
            ax.text((lo + hi) / 2, 0.97, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color="#444444")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB / Hz)")
    ax.set_title(f"{psd_result.label} — Welch PSD (Δf = {psd_result.resolution:.2g} Hz)")
    ax.set_xlim(0, f[-1])

    if standalone:
        save_or_show(fig, save_path)
    return fig


@styled
def plot_hilbert(
    data_original: np.ndarray,
    hilbert_result,
    fs: float = 1.0,
    save_path: str | None = None,
) -> plt.Figure:
    """Signal + envelope, instantaneous phase, instantaneous frequency."""
    fig = plt.figure(figsize=(13, 8))
    gs = gridspec.GridSpec(3, 1, hspace=0.45)
    t = np.arange(len(data_original)) / fs

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, data_original, color=COLORS["primary"], lw=1.0, alpha=0.6, label="signal")
    ax1.plot(t, hilbert_result.envelope, color=COLORS["secondary"], lw=1.8, label="envelope")
    ax1.plot(t, -hilbert_result.envelope, color=COLORS["secondary"], lw=1.2, ls="--", alpha=0.5)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"{hilbert_result.label} — Hilbert transform")
    ax1.legend(fontsize=9, frameon=False)
    ax1.set_xlim(t[0], t[-1])

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t, np.angle(hilbert_result.analytic), color=COLORS["accent"], lw=0.8)
    ax2.set_ylabel("Phase (rad)")
    ax2.set_title("Instantaneous phase (wrapped)")

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    freq = hilbert_result.frequency
    lo, hi = np.percentile(freq, [2, 98])          # phase jumps cause spikes
    ax3.plot(t, np.clip(freq, lo, hi), color=COLORS["warning"], lw=1.2)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Frequency (Hz)")
    ax3.set_title("Instantaneous frequency")

    save_or_show(fig, save_path)
    return fig


@styled
def plot_wavelet(wavelet_result, save_path: str | None = None, cmap: str = "magma") -> plt.Figure:
    """CWT scalogram (dB), log-frequency axis."""
    fig, ax = plt.subplots(figsize=(13, 5))
    power_db = 10 * np.log10(wavelet_result.scalogram() + 1e-12)
    vmax = np.percentile(power_db, 99.5)
    im = ax.pcolormesh(wavelet_result.times, wavelet_result.freqs, power_db,
                       cmap=cmap, shading="gouraud", vmin=vmax - 60, vmax=vmax)
    fig.colorbar(im, ax=ax, pad=0.02).set_label("Power (dB)", fontsize=10)
    f = wavelet_result.freqs
    if f[0] > 0 and f[-1] / f[0] > 10:
        ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{wavelet_result.label} — CWT scalogram")
    save_or_show(fig, save_path)
    return fig


@styled
def plot_stft(stft_result, save_path: str | None = None, cmap: str = "viridis") -> plt.Figure:
    """STFT spectrogram (dB) with a 60 dB dynamic range."""
    fig, ax = plt.subplots(figsize=(13, 5))
    spec_db = stft_result.spectrogram()
    vmax = np.percentile(spec_db, 99.5)
    im = ax.pcolormesh(stft_result.times, stft_result.freqs, spec_db,
                       cmap=cmap, shading="gouraud", vmin=vmax - 60, vmax=vmax)
    fig.colorbar(im, ax=ax, pad=0.02).set_label("Power (dB)", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{stft_result.label} — STFT spectrogram")
    save_or_show(fig, save_path)
    return fig


def _style_table(tbl, header_color: str) -> None:
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (row, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f0f0")


@styled
def plot_descriptive(descriptive_result, data: np.ndarray, save_path: str | None = None) -> plt.Figure:
    """Histogram + KDE, boxplot, Q-Q plot and a stats table."""
    from scipy import stats as sp_stats

    fig = plt.figure(figsize=(13, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)
    r = descriptive_result
    x = np.asarray(data, dtype=float).flatten()
    x = x[~np.isnan(x)]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(x, bins=40, color=COLORS["primary"], alpha=0.5, density=True, label="data")
    if np.ptp(x) > 0:
        kde_x = np.linspace(x.min(), x.max(), 300)
        ax1.plot(kde_x, sp_stats.gaussian_kde(x)(kde_x), color=COLORS["primary"], lw=2, label="KDE")
    ax1.axvline(r.mean, color=COLORS["accent"], lw=1.5, ls="--", label=f"mean={r.mean:.3f}")
    ax1.axvline(r.median, color=COLORS["secondary"], lw=1.5, ls=":", label=f"median={r.median:.3f}")
    ax1.set_title("Distribution")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=8, frameon=False)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.boxplot(
        x, patch_artist=True, widths=0.5,
        boxprops=dict(facecolor=COLORS["primary"], alpha=0.4),
        medianprops=dict(color=COLORS["accent"], linewidth=2),
        whiskerprops=dict(color=COLORS["neutral"]),
        capprops=dict(color=COLORS["neutral"]),
        flierprops=dict(marker="o", markerfacecolor=COLORS["accent"], markersize=3, alpha=0.5),
    )
    ax2.set_title("Boxplot")
    ax2.set_ylabel("Value")
    ax2.set_xticks([])

    ax3 = fig.add_subplot(gs[1, 0])
    (osm, osr), (slope, intercept, _) = sp_stats.probplot(x, dist="norm")
    ax3.scatter(osm, osr, color=COLORS["primary"], s=8, alpha=0.5)
    ax3.plot(osm, slope * osm + intercept, color=COLORS["accent"], lw=1.5, label="normal ref")
    ax3.set_title("Q-Q plot (normality)")
    ax3.set_xlabel("Theoretical quantiles")
    ax3.set_ylabel("Sample quantiles")
    ax3.legend(fontsize=8, frameon=False)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    rows = [["N", f"{r.n:,}"], ["Mean", f"{r.mean:.4f}"], ["Median", f"{r.median:.4f}"],
            ["Std", f"{r.std:.4f}"], ["MAD", f"{r.mad:.4f}"], ["Skewness", f"{r.skewness:.4f}"],
            ["Kurtosis", f"{r.kurtosis:.4f}"], ["Min / Max", f"{r.minimum:.3f} / {r.maximum:.3f}"],
            ["IQR", f"{r.iqr:.4f}"], ["RMS", f"{r.rms:.4f}"], ["SNR (dB)", f"{r.snr:.2f}"]]
    _style_table(ax4.table(cellText=rows, colLabels=["Statistic", "Value"],
                           bbox=[0, 0, 1, 0.92], cellLoc="left"), COLORS["primary"])
    ax4.set_title(f"Stats — '{r.label}'", pad=12)

    save_or_show(fig, save_path)
    return fig


@styled
def plot_correlation_matrix(correlation_result, save_path: str | None = None) -> plt.Figure:
    """Correlation heatmap with annotated values."""
    fig, ax = plt.subplots(figsize=(8, 7))
    mat, labels = correlation_result.matrix, correlation_result.labels
    n = len(labels)
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation coefficient")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(val) > 0.6 else "#333333")
    ax.set_title(f"Correlation matrix ({correlation_result.method})", pad=14)
    save_or_show(fig, save_path)
    return fig


@styled
def plot_autocorrelation(
    autocorr: np.ndarray,
    fs: float = 1.0,
    label: str = "",
    save_path: str | None = None,
    n: int | None = None,
) -> plt.Figure:
    """
    Autocorrelation with ±1.96/√N white-noise bands. N must be the number of
    *signal samples* (not lags) — pass ``n``; it defaults to the lag count
    only for backwards compatibility.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    lags = np.arange(len(autocorr))
    conf = 1.96 / np.sqrt(n or len(autocorr))
    ax.bar(lags / fs, autocorr, width=0.8 / fs, color=COLORS["primary"], alpha=0.6)
    ax.axhline(0, color=COLORS["neutral"], lw=0.8)
    ax.axhline(conf, color=COLORS["accent"], lw=1.2, ls="--", label=f"95% white-noise band (±{conf:.3f})")
    ax.axhline(-conf, color=COLORS["accent"], lw=1.2, ls="--")
    ax.set_xlabel("Lag (s)" if fs != 1.0 else "Lag (samples)")
    ax.set_ylabel("Correlation")
    ax.set_title(f"{label} — autocorrelation")
    ax.legend(fontsize=9, frameon=False)
    save_or_show(fig, save_path)
    return fig


@styled
def plot_quality(report, save_path: str | None = None) -> plt.Figure:
    """
    Visual signal diagnosis: the signal with problem regions highlighted
    (gaps, flatlines, clipping, spikes) and — when fs is known — the PSD
    with the detected line-noise frequency marked.
    """
    from scipy.signal import welch

    x = report.signal
    fs = report.fs or 1.0
    t = np.arange(len(x)) / fs
    has_psd = report.fs is not None and np.isfinite(x).sum() > 16

    fig = plt.figure(figsize=(13, 7 if has_psd else 4))
    gs = gridspec.GridSpec(2 if has_psd else 1, 1, hspace=0.45)
    ax = fig.add_subplot(gs[0])
    ax.plot(t, x, color=COLORS["primary"], lw=0.9, alpha=0.9)

    shade = {"missing": ("#999999", "missing"), "flatline": (COLORS["warning"], "flatline"),
             "clipping": (COLORS["accent"], "clipping")}
    from PBstats.diagnostics.quality import _runs
    for key, (color, name) in shade.items():
        mask = report.masks.get(key)
        if mask is None or not mask.any():
            continue
        for k, (a, b) in enumerate(_runs(mask)):
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color=color, alpha=0.25,
                       label=name if k == 0 else None)
    spikes = report.masks.get("spikes")
    if spikes is not None and spikes.any():
        ax.scatter(t[spikes], x[spikes], color="#C0392B", s=28, zorder=5, label="spikes")

    ax.set_xlabel("Time (s)" if report.fs else "Sample")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Signal diagnosis — '{report.label}'   score {report.score}/100 "
                 f"(grade {report.grade})")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_xlim(t[0], t[-1])

    if has_psd:
        from PBstats.core.utils import interp_nans
        xc = interp_nans(x)
        f, p = welch(xc - xc.mean(), fs=fs, nperseg=int(min(len(xc), max(2 * fs, 64))))
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(f, 10 * np.log10(p + 1e-300), color=COLORS["primary"], lw=1.2)
        if report.has("mains"):
            f0 = report.metrics.get("mains_hz")
            ax2.axvline(f0, color=COLORS["accent"], ls="--", lw=1.2, label=f"{f0:g} Hz line noise")
            ax2.legend(fontsize=9, frameon=False)
        if report.has("drift"):
            ax2.axvspan(0, 0.5, color=COLORS["warning"], alpha=0.2)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("PSD (dB / Hz)")
        ax2.set_title("Power spectral density")
        ax2.set_xlim(0, f[-1])

    save_or_show(fig, save_path)
    return fig


@styled
def plot_epochs(epochs, save_path: str | None = None, max_traces: int = 50) -> plt.Figure:
    """Individual epochs (thin, first channel) with their mean ± SD."""
    data = epochs.data if epochs.data.ndim == 2 else epochs.data[:, :, 0]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    t = epochs.times
    for row in data[:max_traces]:
        ax.plot(t, row, color=COLORS["neutral"], lw=0.5, alpha=0.35)
    if len(data):
        mu, sd = data.mean(axis=0), data.std(axis=0)
        ax.fill_between(t, mu - sd, mu + sd, color=COLORS["primary"], alpha=0.18, label="± SD")
        ax.plot(t, mu, color=COLORS["primary"], lw=2.2, label=f"mean (n={len(data)})")
    ax.axvline(0, color=COLORS["accent"], ls="--", lw=1, label="event")
    ax.set_xlabel("Time relative to event (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{epochs.label} — epochs")
    ax.legend(fontsize=9, frameon=False)
    ax.set_xlim(t[0], t[-1])
    save_or_show(fig, save_path)
    return fig


@styled
def plot_forecast(result, save_path: str | None = None) -> plt.Figure:
    """History, forecast with prediction interval, and held-out truth if any."""
    fig, ax = plt.subplots(figsize=(12, 4))
    n_hist = min(len(result.history), max(5 * len(result.mean), 200))
    ax.plot(result.history_times[-n_hist:], result.history[-n_hist:],
            color=COLORS["primary"], lw=1.2, label="history")
    ax.fill_between(result.times, result.lower, result.upper, color=COLORS["secondary"],
                    alpha=0.2, label=f"{result.level:.0%} interval")
    ax.plot(result.times, result.mean, color=COLORS["secondary"], lw=2, label="forecast")
    if result.actual is not None:
        ax.plot(result.times, result.actual, color=COLORS["accent"], lw=1.2, ls="--", label="actual")
    order = f"ARIMA{result.order}"
    ax.set_title(f"{result.label} — {order} forecast")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.legend(fontsize=9, frameon=False)
    save_or_show(fig, save_path)
    return fig


@styled
def plot_events(data, fs=None, change_points=None, anomalies=None, label: str = "",
                save_path: str | None = None) -> plt.Figure:
    """Signal with change points (vertical lines) and anomalous windows (shaded)."""
    fs = fs or 1.0
    x = np.asarray(data)
    t = np.arange(len(x)) / fs
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(t, x, color=COLORS["primary"], lw=0.8)
    if anomalies is not None and len(anomalies):
        for k, row in enumerate(anomalies[anomalies["anomaly"]].itertuples()):
            ax.axvspan(row.t_start, row.t_end, color=COLORS["accent"], alpha=0.18,
                       label="anomalous window" if k == 0 else None)
    if change_points is not None:
        for k, c in enumerate(change_points.times):
            ax.axvline(c, color=COLORS["warning"], lw=1.6, ls="--",
                       label="change point" if k == 0 else None)
        for seg in change_points.segments.itertuples():
            ax.hlines(seg.mean, seg.start_s, seg.end_s, color=COLORS["secondary"], lw=2)
    ax.set_xlabel("Time (s)" if fs != 1.0 else "Sample")
    ax.set_title(f"{label} — events")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_xlim(t[0], t[-1])
    save_or_show(fig, save_path)
    return fig


@styled
def plot_quality_timeline(timeline, label: str = "", save_path: str | None = None) -> plt.Figure:
    """Quality score per window as a coloured strip — where is the recording bad?"""
    fig, ax = plt.subplots(figsize=(13, 2.6))
    colors = np.where(timeline["score"] >= 90, COLORS["secondary"],
                      np.where(timeline["score"] >= 60, COLORS["warning"], COLORS["accent"]))
    ax.bar(timeline["t_start"], timeline["score"], width=timeline["t_end"] - timeline["t_start"],
           align="edge", color=colors, edgecolor="white", linewidth=0.3)
    ax.set_ylim(0, 100)
    ax.set_xlim(timeline["t_start"].iloc[0], timeline["t_end"].iloc[-1])
    ax.set_ylabel("Score")
    ax.set_xlabel("Time (s)")
    ok = timeline["ok"].mean()
    ax.set_title(f"{label} — quality over time ({ok:.0%} of windows OK)")
    save_or_show(fig, save_path)
    return fig
