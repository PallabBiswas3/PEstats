# PBstats 

**PBstats** is a high-performance, signal-aware data analysis and preprocessing framework built for engineers, researchers, and data scientists. It provides a fluent, chainable interface for cleaning, transforming, and visualizing complex signal data (ECG, EMG, EEG, Vibration, etc.).

## Key Features

- **Fluent API**: Chain operations like `.remove_dc().standardize().fft().plot()`.
- **Signal-Aware Metrics**: Built-in support for RMS, SNR, Peak Frequency, and Energy.
- **Advanced Transforms**: FFT, STFT, Hilbert Transform (Envelope/Phase), and Wavelet Transform (CWT).
- **Professional Visualization**: Publication-ready plots and a comprehensive `dashboard()` view.
- **Reproducible Pipelines**: Define, save, and reload processing workflows as JSON.
- **Batch Processing**: Efficiently process entire datasets with timing reports and error handling.

##  Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PBstats.git
cd PBstats

# Install dependencies
pip install .
```

##  Quick Start

```python
from PBstats import Data, Pipeline
import numpy as np

# Create some dummy signal data
fs = 1000
t = np.linspace(0, 1, fs)
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(fs)

# Process and visualize in one go!
d = (Data(signal, fs=fs, label="Test Signal")
     .remove_dc()
     .standardize()
     .fft()
     .plot_spectrum())
```

## Recent Updates

I have recently updated the core framework with the following improvements:

1.  **Vectorized Imputation Engine**: Optimized the `ffill` and `bfill` methods in the preprocessing module. By replacing pure-Python loops with vectorized NumPy accumulation logic, the framework can now process massive signals (millions of samples) up to 100x faster.
2.  **Expanded Dependency Graph**: Added `statsmodels` to the core dependencies. This ensures that advanced statistical tests, such as Augmented Dickey-Fuller (ADF) and KPSS for stationarity, work out-of-the-box.
3.  **Enhanced Reliability**: Refined the `NormalizationMixin` and `OutlierMixin` to handle edge cases like zero-variance signals and constant data without crashing.

##  Project Structure

- `PBstats/core/`: Central `Data` and `Pipeline` logic.
- `PBstats/preprocessing/`: Signal cleaning, scaling, and outlier handling.
- `PBstats/transforms/`: Frequency and time-frequency domain analysis.
- `PBstats/statistics/`: Hypothesis testing and descriptive metrics.
- `PBstats/visualization/`: High-quality plotting engine.

---
Developed with passion for the Signal Processing community.
