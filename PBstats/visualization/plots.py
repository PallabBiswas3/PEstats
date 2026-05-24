from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PBstats.visualization.base import apply_style, COLORS, save_or_show


def plot_signal(
    data: np.ndarray,
    fs: float = 1.0,
    label: str = "signal",
    save_path: str = None,
    ax: plt.Axes = None,
) -> plt.Figure:
    """
    Plot a raw time-domain signal.
    If ax is provided, draws into that axes (for subplots).
    """
    apply_style()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 3))
    else:
        fig = ax.get_figure()

    t = np.arange(len(data)) / fs
    ax.plot(t, data, color=COLORS["primary"], lw=1.4, alpha=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{label} — time domain")
    ax.set_xlim(t[0], t[-1])

    if standalone:
        save_or_show(fig, save_path)
    return fig


def plot_fft(
    fft_result,
    max_freq: float = None,
    log_scale: bool = False,
    mark_peaks: int = 3,
    save_path: str = None,
    ax: plt.Axes = None,
) -> plt.Figure:
    """
    Plot FFT magnitude spectrum with optional peak markers.

    Parameters
    ----------
    max_freq   : clip x-axis at this frequency (Hz)
    log_scale  : use log scale on y-axis
    mark_peaks : highlight the N strongest frequency components
    """
    apply_style()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.get_figure()

    freqs = fft_result.freqs
    mag   = fft_result.magnitude

    # Clip to max_freq
    if max_freq:
        mask  = freqs <= max_freq
        freqs = freqs[mask]
        mag   = mag[mask]

    ax.fill_between(freqs, mag, alpha=0.15, color=COLORS["primary"])
    ax.plot(freqs, mag, color=COLORS["primary"], lw=1.4)

    if log_scale:
        ax.set_yscale("log")

    # Mark dominant peaks
    if mark_peaks > 0:
        peak_idx = np.argsort(mag)[::-1][:mark_peaks]
        for idx in peak_idx:
            ax.axvline(freqs[idx], color=COLORS["accent"],
                       lw=1.0, ls="--", alpha=0.7)
            ax.annotate(
                f"{freqs[idx]:.1f} Hz",
                xy=(freqs[idx], mag[idx]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=9, color=COLORS["accent"],
            )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"{fft_result.label} — frequency spectrum")
    ax.set_xlim(0, freqs[-1])

    if standalone:
        save_or_show(fig, save_path)
    return fig


def plot_hilbert(
    data_original: np.ndarray,
    hilbert_result,
    fs: float = 1.0,
    save_path: str = None,
) -> plt.Figure:
    """
    Three-panel Hilbert transform plot:
    top    — original signal + envelope overlay
    middle — instantaneous phase
    bottom — instantaneous frequency
    """
    apply_style()
    fig = plt.figure(figsize=(13, 8))
    gs  = gridspec.GridSpec(3, 1, hspace=0.45)

    t = np.arange(len(data_original)) / fs

    # Panel 1 — signal + envelope
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, data_original,
             color=COLORS["primary"],  lw=1.2, alpha=0.6, label="signal")
    ax1.plot(t, hilbert_result.envelope,
             color=COLORS["secondary"], lw=1.8, label="envelope")
    ax1.plot(t, -hilbert_result.envelope,
             color=COLORS["secondary"], lw=1.8, ls="--", alpha=0.5)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"{hilbert_result.label} — Hilbert transform")
    ax1.legend(fontsize=9, frameon=False)
    ax1.set_xlim(t[0], t[-1])

    # Panel 2 — instantaneous phase
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t, hilbert_result.phase,
             color=COLORS["accent"], lw=1.2)
    ax2.set_ylabel("Phase (rad)")
    ax2.set_title("Instantaneous phase")
    ax2.set_xlim(t[0], t[-1])

    # Panel 3 — instantaneous frequency
    ax3 = fig.add_subplot(gs[2])
    # Clip extreme values for clean plot (frequency can spike at phase jumps)
    freq_clipped = np.clip(
        hilbert_result.frequency,
        np.percentile(hilbert_result.frequency, 2),
        np.percentile(hilbert_result.frequency, 98),
    )
    ax3.plot(t, freq_clipped,
             color=COLORS["warning"], lw=1.2)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Frequency (Hz)")
    ax3.set_title("Instantaneous frequency")
    ax3.set_xlim(t[0], t[-1])

    save_or_show(fig, save_path)
    return fig


def plot_wavelet(
    wavelet_result,
    save_path: str = None,
    cmap: str = "magma",
) -> plt.Figure:
    """
    Plot a CWT scalogram (time-frequency power heatmap).
    Warmer colors = higher power at that time-frequency point.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(13, 5))

    power = wavelet_result.scalogram()
    power_db = 10 * np.log10(power + 1e-12)   # convert to dB

    im = ax.pcolormesh(
        wavelet_result.times,
        wavelet_result.freqs,
        power_db,
        cmap=cmap,
        shading="gouraud",
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Power (dB)", fontsize=10)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{wavelet_result.label} — CWT scalogram")

    save_or_show(fig, save_path)
    return fig


def plot_stft(
    stft_result,
    save_path: str = None,
    cmap: str = "viridis",
) -> plt.Figure:
    """
    Plot STFT spectrogram — how frequency content evolves over time.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(13, 5))

    spec_db = stft_result.spectrogram()

    im = ax.pcolormesh(
        stft_result.times,
        stft_result.freqs,
        spec_db,
        cmap=cmap,
        shading="gouraud",
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Power (dB)", fontsize=10)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{stft_result.label} — STFT spectrogram")

    save_or_show(fig, save_path)
    return fig


def plot_descriptive(
    descriptive_result,
    data: np.ndarray,
    save_path: str = None,
) -> plt.Figure:
    """
    Four-panel descriptive statistics summary:
    top-left   — histogram + KDE
    top-right  — boxplot
    bottom-left — Q-Q plot (normality check)
    bottom-right — stat summary table
    """
    apply_style()
    from scipy import stats as sp_stats

    fig = plt.figure(figsize=(13, 8))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)
    r   = descriptive_result
    x   = data.flatten()

    # Panel 1 — histogram + KDE
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(x, bins=40, color=COLORS["primary"],
             alpha=0.5, density=True, label="data")

    # KDE overlay
    kde_x = np.linspace(x.min(), x.max(), 300)
    kde   = sp_stats.gaussian_kde(x)
    ax1.plot(kde_x, kde(kde_x),
             color=COLORS["primary"], lw=2, label="KDE")
    ax1.axvline(r.mean,   color=COLORS["accent"],
                lw=1.5, ls="--", label=f"mean={r.mean:.3f}")
    ax1.axvline(r.median, color=COLORS["secondary"],
                lw=1.5, ls=":",  label=f"median={r.median:.3f}")
    ax1.set_title("Distribution")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=8, frameon=False)

    # Panel 2 — boxplot
    ax2 = fig.add_subplot(gs[0, 1])
    bp  = ax2.boxplot(
        x,
        vert=True,
        patch_artist=True,
        widths=0.5,
        boxprops    =dict(facecolor=COLORS["primary"], alpha=0.4),
        medianprops =dict(color=COLORS["accent"],   linewidth=2),
        whiskerprops=dict(color=COLORS["neutral"]),
        capprops    =dict(color=COLORS["neutral"]),
        flierprops  =dict(marker="o", color=COLORS["accent"],
                          markersize=3, alpha=0.5),
    )
    ax2.set_title("Boxplot")
    ax2.set_ylabel("Value")
    ax2.set_xticks([])

    # Panel 3 — Q-Q plot
    ax3 = fig.add_subplot(gs[1, 0])
    theoretical, sample = sp_stats.probplot(x, dist="norm")[:2][0], \
                          sp_stats.probplot(x, dist="norm")[0][1]
    prob = sp_stats.probplot(x, dist="norm")
    ax3.scatter(prob[0][0], prob[0][1],
                color=COLORS["primary"], s=8, alpha=0.5)
    ax3.plot(prob[0][0], prob[1][0] * prob[0][0] + prob[1][1],
             color=COLORS["accent"], lw=1.5, label="normal ref")
    ax3.set_title("Q-Q plot (normality)")
    ax3.set_xlabel("Theoretical quantiles")
    ax3.set_ylabel("Sample quantiles")
    ax3.legend(fontsize=8, frameon=False)

    # Panel 4 — stats table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    rows = [
        ["N",          f"{r.n:,}"],
        ["Mean",       f"{r.mean:.4f}"],
        ["Median",     f"{r.median:.4f}"],
        ["Std",        f"{r.std:.4f}"],
        ["Skewness",   f"{r.skewness:.4f}"],
        ["Kurtosis",   f"{r.kurtosis:.4f}"],
        ["Min",        f"{r.minimum:.4f}"],
        ["Max",        f"{r.maximum:.4f}"],
        ["IQR",        f"{r.iqr:.4f}"],
        ["RMS",        f"{r.rms:.4f}"],
        ["SNR (dB)",   f"{r.snr:.2f}"],
    ]
    tbl = ax4.table(
        cellText=rows,
        colLabels=["Statistic", "Value"],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 1.6)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor(COLORS["primary"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f0f0")

    ax4.set_title(f"Stats — '{r.label}'", pad=12)

    save_or_show(fig, save_path)
    return fig


def plot_correlation_matrix(
    correlation_result,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot a correlation heatmap with annotated values.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    mat    = correlation_result.matrix
    labels = correlation_result.labels
    n      = len(labels)

    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation coefficient")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate each cell
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            color = "white" if abs(val) > 0.6 else "#333333"
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=9, color=color, fontweight="500")

    ax.set_title(
        f"Correlation matrix ({correlation_result.method})", pad=14
    )

    save_or_show(fig, save_path)
    return fig


def plot_autocorrelation(
    autocorr: np.ndarray,
    fs: float = 1.0,
    label: str = "",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot autocorrelation function with confidence bands.
    The ±1.96/√N bands show the 95% significance threshold —
    lags outside these bands indicate real periodicity.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 4))

    lags = np.arange(len(autocorr))
    n    = len(autocorr)
    conf = 1.96 / np.sqrt(n)     # 95% confidence band

    ax.bar(lags, autocorr, width=0.8,
           color=COLORS["primary"], alpha=0.6)
    ax.axhline(0,     color=COLORS["neutral"], lw=0.8)
    ax.axhline( conf, color=COLORS["accent"],
                lw=1.2, ls="--", label=f"±95% CI ({conf:.3f})")
    ax.axhline(-conf, color=COLORS["accent"], lw=1.2, ls="--")

    ax.set_xlabel("Lag (samples)")
    ax.set_ylabel("Correlation")
    ax.set_title(f"{label} — autocorrelation")
    ax.legend(fontsize=9, frameon=False)

    # Second x-axis in seconds
    ax2 = ax.twiny()
    ax2.set_xlim(0, (len(autocorr) - 1) / fs)
    ax2.set_xlabel("Lag (s)")

    save_or_show(fig, save_path)
    return fig