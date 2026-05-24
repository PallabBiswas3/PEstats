from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from PBstats.visualization import plots


class VisualizationMixin:
    """
    Attaches .plot_*() methods directly to Data.
    All methods are non-destructive and return self for chaining.
    """

    def plot(self, save_path: str = None) -> "VisualizationMixin":
        """Plot current self.data as a time-domain signal."""
        plots.plot_signal(
            self.data, fs=self.fs or 1.0,
            label=self.label, save_path=save_path,
        )
        return self

    def plot_spectrum(
        self,
        max_freq: float = None,
        mark_peaks: int = 3,
        save_path: str = None,
    ) -> "VisualizationMixin":
        """Plot FFT spectrum. Requires .fft() to have been called."""
        if not hasattr(self, "fft_result"):
            raise RuntimeError(
                "No FFT result found. Call .fft() before .plot_spectrum()."
            )
        plots.plot_fft(
            self.fft_result,
            max_freq=max_freq,
            mark_peaks=mark_peaks,
            save_path=save_path,
        )
        return self

    def plot_envelope(self, save_path: str = None) -> "VisualizationMixin":
        """Plot Hilbert envelope. Requires .hilbert() to have been called."""
        if not hasattr(self, "hilbert_result"):
            raise RuntimeError(
                "No Hilbert result found. Call .hilbert() before .plot_envelope()."
            )
        plots.plot_hilbert(
            self._raw, self.hilbert_result,
            fs=self.fs or 1.0, save_path=save_path,
        )
        return self

    def plot_scalogram(
        self, save_path: str = None, cmap: str = "magma"
    ) -> "VisualizationMixin":
        """Plot CWT scalogram. Requires .wavelet() to have been called."""
        if not hasattr(self, "wavelet_result"):
            raise RuntimeError(
                "No wavelet result found. Call .wavelet() before .plot_scalogram()."
            )
        plots.plot_wavelet(
            self.wavelet_result, save_path=save_path, cmap=cmap
        )
        return self

    def plot_spectrogram(
        self, save_path: str = None, cmap: str = "viridis"
    ) -> "VisualizationMixin":
        """Plot STFT spectrogram. Requires .stft() to have been called."""
        if not hasattr(self, "stft_result"):
            raise RuntimeError(
                "No STFT result. Call .stft() before .plot_spectrogram()."
            )
        plots.plot_stft(
            self.stft_result, save_path=save_path, cmap=cmap
        )
        return self

    def plot_stats(self, save_path: str = None) -> "VisualizationMixin":
        """Four-panel stats summary. Requires .describe() to have been called."""
        if not hasattr(self, "descriptive_result"):
            raise RuntimeError(
                "No descriptive result. Call .describe() before .plot_stats()."
            )
        plots.plot_descriptive(
            self.descriptive_result, self._raw, save_path=save_path
        )
        return self

    def plot_autocorr(self, save_path: str = None) -> "VisualizationMixin":
        """Plot autocorrelation. Requires .autocorrelation() to have been called."""
        if not hasattr(self, "autocorr"):
            raise RuntimeError(
                "No autocorr result. Call .autocorrelation() before .plot_autocorr()."
            )
        plots.plot_autocorrelation(
            self.autocorr, fs=self.fs or 1.0,
            label=self.label, save_path=save_path,
        )
        return self

    def plot_correlation(self, save_path: str = None) -> "VisualizationMixin":
        """Plot correlation heatmap. Requires .correlate() to have been called."""
        if not hasattr(self, "correlation_result"):
            raise RuntimeError(
                "No correlation result. Call .correlate() before .plot_correlation()."
            )
        plots.plot_correlation_matrix(
            self.correlation_result, save_path=save_path
        )
        return self

    def dashboard(self, save_path: str = None) -> "VisualizationMixin":
        """
        Full 6-panel analysis dashboard in one figure.
        Runs whatever results are available — skips panels
        for transforms that haven't been called yet.
        """
        from PBstats.visualization.base import apply_style, save_or_show
        import matplotlib.gridspec as gridspec

        apply_style()

        has_fft     = hasattr(self, "fft_result")
        has_hilbert = hasattr(self, "hilbert_result")
        has_stats   = hasattr(self, "descriptive_result")

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(
            f"pallabstats dashboard — '{self.label}'",
            fontsize=14, fontweight="500", y=1.01,
        )
        gs = gridspec.GridSpec(3, 2, hspace=0.55, wspace=0.35)

        # Row 1 left — raw signal
        ax1 = fig.add_subplot(gs[0, 0])
        plots.plot_signal(
            self._raw, fs=self.fs or 1.0,
            label="raw signal", ax=ax1,
        )

        # Row 1 right — current (processed) signal
        ax2 = fig.add_subplot(gs[0, 1])
        plots.plot_signal(
            self.data, fs=self.fs or 1.0,
            label="processed", ax=ax2,
        )

        # Row 2 left — FFT if available
        ax3 = fig.add_subplot(gs[1, 0])
        if has_fft:
            plots.plot_fft(self.fft_result, mark_peaks=3, ax=ax3)
        else:
            ax3.text(0.5, 0.5, "FFT not computed\ncall .fft()",
                     ha="center", va="center", transform=ax3.transAxes,
                     color="#aaaaaa", fontsize=11)
            ax3.set_title("Frequency spectrum")

        # Row 2 right — envelope if available
        ax4 = fig.add_subplot(gs[1, 1])
        if has_hilbert:
            t = np.arange(len(self._raw)) / (self.fs or 1.0)
            ax4.plot(t, self._raw,
                     color="#5B4FCF", lw=1.0, alpha=0.5, label="signal")
            ax4.plot(t, self.hilbert_result.envelope,
                     color="#1D9E75", lw=2.0, label="envelope")
            ax4.set_title(f"{self.label} — envelope")
            ax4.set_xlabel("Time (s)")
            ax4.legend(fontsize=8, frameon=False)
        else:
            ax4.text(0.5, 0.5, "Hilbert not computed\ncall .hilbert()",
                     ha="center", va="center", transform=ax4.transAxes,
                     color="#aaaaaa", fontsize=11)
            ax4.set_title("Amplitude envelope")

        # Row 3 left — histogram
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(self._raw.flatten(), bins=40,
                 color="#5B4FCF", alpha=0.5, density=True)
        ax5.set_title("Raw signal distribution")
        ax5.set_xlabel("Value")
        ax5.set_ylabel("Density")

        # Row 3 right — stats table or placeholder
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis("off")
        if has_stats:
            r = self.descriptive_result
            rows = [
                ["Mean",     f"{r.mean:.4f}"],
                ["Std",      f"{r.std:.4f}"],
                ["Skew",     f"{r.skewness:.4f}"],
                ["Kurt",     f"{r.kurtosis:.4f}"],
                ["RMS",      f"{r.rms:.4f}"],
                ["SNR (dB)", f"{r.snr:.2f}"],
            ]
            if has_fft:
                rows.append(["Peak freq",
                              f"{self.fft_result.peak_frequency():.1f} Hz"])
            if has_hilbert:
                rows.append(["Peak env",
                              f"{self.hilbert_result.peak_envelope():.4f}"])

            tbl = ax6.table(
                cellText=rows,
                colLabels=["Metric", "Value"],
                loc="center", cellLoc="left",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1.1, 1.7)
            for (row, col), cell in tbl.get_celld().items():
                cell.set_edgecolor("#dddddd")
                if row == 0:
                    cell.set_facecolor("#5B4FCF")
                    cell.set_text_props(color="white", fontweight="bold")
                elif row % 2 == 0:
                    cell.set_facecolor("#f4f4f4")
        else:
            ax6.text(0.5, 0.5, "Call .describe() for stats",
                     ha="center", va="center", transform=ax6.transAxes,
                     color="#aaaaaa", fontsize=11)
        ax6.set_title("Analysis summary")

        save_or_show(fig, save_path)
        return self