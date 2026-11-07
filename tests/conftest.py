import matplotlib

matplotlib.use("Agg")  # headless: plots render to files, never block

import numpy as np
import pytest


def synth_ecg(fs=500, dur=20.0, hr=70, seed=0, noise=0.01):
    """Synthetic ECG: P, Q, R, S, T Gaussians per beat with ~5% RR variability."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * dur)) / fs
    x = np.zeros_like(t)
    beat = 0.5
    waves = [(0.12, -0.2, 0.025), (-0.1, -0.03, 0.008), (1.0, 0.0, 0.01),
             (-0.15, 0.03, 0.01), (0.3, 0.25, 0.04)]
    while beat < dur:
        for a, mu, sig in waves:
            x += a * np.exp(-((t - beat - mu) ** 2) / (2 * sig ** 2))
        beat += 60 / hr * (1 + 0.05 * rng.standard_normal())
    return t, x + noise * rng.standard_normal(len(t))


@pytest.fixture
def ecg():
    return synth_ecg
