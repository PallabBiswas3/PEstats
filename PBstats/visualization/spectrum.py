from __future__ import annotations

import numpy as np

from PBstats.visualization import plots


class VisualizationMixin:
    """
    ``.plot_*()`` methods on Data. All are non-destructive and return self.
    Pass ``save_path=`` to write a file instead of showing the figure.
    """

    def _need(self, attr: str, call: str, plot: str) -> None:
        if not hasattr(self, attr):
            raise RuntimeError(f"Nothing to plot — call .{call}() before .{plot}().")

    def plot(self, save_path: str | None = None) -> "VisualizationMixin":
        """Plot the current data in the time domain (all channels)."""
        plots.plot_signal(self.data, fs=self.fs or 1.0, label=self.label,
                          save_path=save_path, channel_names=self.channel_names)
        return self

    def plot_spectrum(
        self, max_freq: float | None = None, mark_peaks: int = 3, save_path: str | None = None,
    ) -> "VisualizationMixin":
        """FFT spectrum. Requires .fft()."""
        self._need("fft_result", "fft", "plot_spectrum")
        plots.plot_fft(self.fft_result, max_freq=max_freq, mark_peaks=mark_peaks,
                       save_path=save_path)
        return self

    def plot_psd(
        self, max_freq: float | None = None, bands: dict | None = None,
        save_path: str | None = None,
    ) -> "VisualizationMixin":
        """Welch PSD. Requires .psd(). bands e.g. {"alpha": (8, 13)}."""
        self._need("psd_result", "psd", "plot_psd")
        plots.plot_psd(self.psd_result, max_freq=max_freq, bands=bands, save_path=save_path)
        return self

    def plot_envelope(self, save_path: str | None = None) -> "VisualizationMixin":
        """Hilbert envelope over the exact signal that was transformed. Requires .hilbert()."""
        self._need("hilbert_result", "hilbert", "plot_envelope")
        r = self.hilbert_result
        signal = r.signal if r.signal is not None else self._raw
        plots.plot_hilbert(signal, r, fs=r.fs or 1.0, save_path=save_path)
        return self

    def plot_scalogram(self, save_path: str | None = None, cmap: str = "magma") -> "VisualizationMixin":
        """CWT scalogram. Requires .wavelet()."""
        self._need("wavelet_result", "wavelet", "plot_scalogram")
        plots.plot_wavelet(self.wavelet_result, save_path=save_path, cmap=cmap)
        return self

    def plot_spectrogram(self, save_path: str | None = None, cmap: str = "viridis") -> "VisualizationMixin":
        """STFT spectrogram. Requires .stft()."""
        self._need("stft_result", "stft", "plot_spectrogram")
        plots.plot_stft(self.stft_result, save_path=save_path, cmap=cmap)
        return self

    def plot_stats(self, save_path: str | None = None) -> "VisualizationMixin":
        """Four-panel stats summary of the data that was described. Requires .describe()."""
        self._need("descriptive_result", "describe", "plot_stats")
        r = self.descriptive_result
        values = r.values if r.values is not None else self.data
        plots.plot_descriptive(r, values, save_path=save_path)
        return self

    def plot_autocorr(self, save_path: str | None = None) -> "VisualizationMixin":
        """Autocorrelation with significance bands. Requires .autocorrelation()."""
        self._need("autocorr", "autocorrelation", "plot_autocorr")
        plots.plot_autocorrelation(self.autocorr, fs=self.fs or 1.0, label=self.label,
                                   save_path=save_path, n=getattr(self, "autocorr_n", None))
        return self

    def plot_correlation(self, save_path: str | None = None) -> "VisualizationMixin":
        """Correlation heatmap. Requires .correlate()."""
        self._need("correlation_result", "correlate", "plot_correlation")
        plots.plot_correlation_matrix(self.correlation_result, save_path=save_path)
        return self

    def plot_quality(self, save_path: str | None = None) -> "VisualizationMixin":
        """Visual signal diagnosis. Runs .diagnose() if needed."""
        if not hasattr(self, "quality_report"):
            self.diagnose()
        plots.plot_quality(self.quality_report, save_path=save_path)
        return self

    def plot_quality_timeline(self, window: float = 10.0, save_path: str | None = None,
                              **diagnose_kwargs) -> "VisualizationMixin":
        """Quality score per window. Runs .quality_timeline() if needed."""
        tl = getattr(self, "quality_timeline_result", None)
        if tl is None:
            tl = self.quality_timeline(window=window, **diagnose_kwargs)
        plots.plot_quality_timeline(tl, self.label, save_path=save_path)
        return self

    def plot_events(self, save_path: str | None = None) -> "VisualizationMixin":
        """Signal with detected change points and anomalous windows.
        Requires .detect_change_points() and/or .anomaly_scores()."""
        cp = getattr(self, "change_point_result", None)
        an = getattr(self, "anomaly_result", None)
        if cp is None and an is None:
            raise RuntimeError("Call .detect_change_points() or .anomaly_scores() before .plot_events().")
        plots.plot_events(self.data, self.fs, cp, an, self.label, save_path=save_path)
        return self

    def plot_forecast(self, save_path: str | None = None) -> "VisualizationMixin":
        """History + forecast with prediction interval. Requires .forecast()."""
        self._need("forecast_result", "forecast", "plot_forecast")
        plots.plot_forecast(self.forecast_result, save_path=save_path)
        return self

    def dashboard(self, save_path: str | None = None) -> "VisualizationMixin":
        """
        Six-panel overview: raw vs processed signal, spectrum, envelope,
        distribution and a summary table. Panels for analyses not yet run
        show a hint instead.
        """
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        from PBstats.visualization.base import COLORS, save_or_show, style

        fs = self.fs or 1.0
        raw_fs = self._raw_fs or 1.0
        hint = dict(ha="center", va="center", color="#aaaaaa", fontsize=11)

        def placeholder(ax, text, title):
            ax.text(0.5, 0.5, text, transform=ax.transAxes, **hint)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)

        with style():
            fig = plt.figure(figsize=(16, 10))
            fig.suptitle(f"PBstats dashboard — '{self.label}'", fontsize=14, fontweight="bold")
            gs = gridspec.GridSpec(3, 2, hspace=0.6, wspace=0.3)

            plots.plot_signal(self._raw, fs=raw_fs, label="raw", ax=fig.add_subplot(gs[0, 0]),
                              channel_names=self.channel_names)
            ax2 = fig.add_subplot(gs[0, 1])
            if hasattr(self, "fft_result") or hasattr(self, "hilbert_result"):
                placeholder(ax2, "data is now in a transform domain\n(see panels below)", "processed")
            else:
                plots.plot_signal(self.data, fs=fs, label="processed", ax=ax2,
                                  channel_names=self.channel_names)

            ax3 = fig.add_subplot(gs[1, 0])
            if hasattr(self, "psd_result"):
                plots.plot_psd(self.psd_result, ax=ax3)
            elif hasattr(self, "fft_result"):
                plots.plot_fft(self.fft_result, mark_peaks=3, ax=ax3)
            else:
                placeholder(ax3, "Spectrum not computed\ncall .psd() or .fft()", "Frequency spectrum")

            ax4 = fig.add_subplot(gs[1, 1])
            if hasattr(self, "hilbert_result"):
                r = self.hilbert_result
                t = np.arange(len(r.envelope)) / (r.fs or 1.0)
                ax4.plot(t, r.signal, color=COLORS["primary"], lw=0.9, alpha=0.5, label="signal")
                ax4.plot(t, r.envelope, color=COLORS["secondary"], lw=1.8, label="envelope")
                ax4.set_xlabel("Time (s)")
                ax4.legend(fontsize=8, frameon=False)
                ax4.set_title("Amplitude envelope")
            else:
                placeholder(ax4, "Hilbert not computed\ncall .hilbert()", "Amplitude envelope")

            ax5 = fig.add_subplot(gs[2, 0])
            raw = self._raw.flatten()
            ax5.hist(raw[~np.isnan(raw)], bins=40, color=COLORS["primary"], alpha=0.5, density=True)
            ax5.set_title("Raw signal distribution")
            ax5.set_xlabel("Value")
            ax5.set_ylabel("Density")

            ax6 = fig.add_subplot(gs[2, 1])
            ax6.axis("off")
            rows = []
            if hasattr(self, "quality_report"):
                q = self.quality_report
                after = getattr(self, "quality_report_after", None)
                rows.append(["Quality", f"{q.score}/100 ({q.grade})" if after is None
                             else f"{q.score} → {after.score}/100 (auto_clean)"])
            if hasattr(self, "descriptive_result"):
                r = self.descriptive_result
                rows += [["Mean", f"{r.mean:.4f}"], ["Std", f"{r.std:.4f}"],
                         ["Skew", f"{r.skewness:.4f}"], ["Kurt", f"{r.kurtosis:.4f}"],
                         ["RMS", f"{r.rms:.4f}"], ["SNR (dB)", f"{r.snr:.2f}"]]
            if hasattr(self, "psd_result"):
                rows.append(["Peak freq (PSD)", f"{self.psd_result.peak_frequency(fmin=0.1):.2f} Hz"])
            elif hasattr(self, "fft_result"):
                rows.append(["Peak freq", f"{self.fft_result.peak_frequency():.2f} Hz"])
            if hasattr(self, "hilbert_result"):
                rows.append(["Peak envelope", f"{self.hilbert_result.peak_envelope():.4f}"])
            if rows:
                plots._style_table(ax6.table(cellText=rows, colLabels=["Metric", "Value"],
                                             bbox=[0, 0, 1, 0.9], cellLoc="left"), COLORS["primary"])
            else:
                ax6.text(0.5, 0.5, "Call .describe() / .diagnose() for a summary",
                         transform=ax6.transAxes, **hint)
            ax6.set_title("Analysis summary")

            save_or_show(fig, save_path)
        return self
