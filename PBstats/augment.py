"""
Data augmentation and deep-learning export.

The augmentations deliberately mirror the artefacts that ``diagnose()``
detects in real recordings — sensor noise, power-line hum, baseline wander,
impulse spikes, dropouts, amplitude/gain changes, timing jitter — so a model
can be trained to be robust to exactly what it will meet in the field.

    aug = Augmenter(noise_snr=(10, 30), powerline=0.3, wander=0.3, seed=0)
    X_aug, y_aug = aug.expand(X, y, copies=4)        # 5× the training data

    X, y = make_windows(signals, fs=250, window=10, step=5, labels=labels)
    ds = to_torch_dataset(X, y, augmenter=aug)       # augments on the fly
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# ═════════════════════════════════════════════════════════════════════════
# Single transforms — all operate on the last axis (time); x may be 1D or
# (..., T). They return new arrays.
# ═════════════════════════════════════════════════════════════════════════


def _rng(rng) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def add_noise(x, snr_db: float, rng=None) -> np.ndarray:
    """Add white Gaussian noise at the given signal-to-noise ratio (dB)."""
    x = np.asarray(x, dtype=float)
    p_sig = np.mean(x ** 2, axis=-1, keepdims=True)
    sigma = np.sqrt(p_sig / 10 ** (snr_db / 10))
    return x + sigma * _rng(rng).standard_normal(x.shape)


def add_powerline(x, fs: float, freq: float = 50.0, amplitude: float = 0.1, rng=None) -> np.ndarray:
    """Add mains hum; amplitude relative to the signal's standard deviation."""
    x = np.asarray(x, dtype=float)
    t = np.arange(x.shape[-1]) / fs
    phase = _rng(rng).uniform(0, 2 * np.pi)
    return x + amplitude * np.std(x, axis=-1, keepdims=True) * np.sin(2 * np.pi * freq * t + phase)


def add_baseline_wander(x, fs: float, amplitude: float = 0.5, freq: float = 0.2, rng=None) -> np.ndarray:
    """Add slow sinusoidal drift (breathing / electrode movement)."""
    x = np.asarray(x, dtype=float)
    r = _rng(rng)
    t = np.arange(x.shape[-1]) / fs
    f = freq * r.uniform(0.5, 1.5)
    return x + amplitude * np.std(x, axis=-1, keepdims=True) * np.sin(2 * np.pi * f * t + r.uniform(0, 2 * np.pi))


def add_spikes(x, n: int = 3, amplitude: float = 5.0, rng=None) -> np.ndarray:
    """Add ``n`` single-sample impulses of ±amplitude × std (electrode pops)."""
    x = np.asarray(x, dtype=float).copy()
    r = _rng(rng)
    sd = np.std(x, axis=-1, keepdims=True)
    flat = x.reshape(-1, x.shape[-1])
    sdf = sd.reshape(-1, 1)
    for row in range(flat.shape[0]):
        idx = r.integers(0, flat.shape[1], n)
        flat[row, idx] += r.choice([-1, 1], n) * amplitude * sdf[row, 0]
    return flat.reshape(x.shape)


def dropout(x, max_fraction: float = 0.1, rng=None) -> np.ndarray:
    """Zero (flat-line) one random segment of up to ``max_fraction`` of the length."""
    x = np.asarray(x, dtype=float).copy()
    r = _rng(rng)
    T = x.shape[-1]
    length = int(r.uniform(0, max_fraction) * T)
    if length:
        start = r.integers(0, T - length + 1)
        x[..., start:start + length] = 0.0
    return x


def scale(x, factor: float) -> np.ndarray:
    """Multiply amplitude (sensor gain differences)."""
    return np.asarray(x, dtype=float) * factor


def time_shift(x, shift: int) -> np.ndarray:
    """Circularly shift by ``shift`` samples."""
    return np.roll(np.asarray(x, dtype=float), shift, axis=-1)


def time_stretch(x, factor: float) -> np.ndarray:
    """Speed up (factor > 1) or slow down, keeping the length (crop/edge-pad).
    Mimics heart-rate / cadence variation."""
    x = np.asarray(x, dtype=float)
    T = x.shape[-1]
    src = np.clip(np.arange(T) * factor, 0, T - 1)
    flat = x.reshape(-1, T)
    out = np.stack([np.interp(src, np.arange(T), row) for row in flat])
    return out.reshape(x.shape)


def invert(x) -> np.ndarray:
    """Flip polarity (reversed electrode placement)."""
    return -np.asarray(x, dtype=float)


# ═════════════════════════════════════════════════════════════════════════
# Random composition
# ═════════════════════════════════════════════════════════════════════════


class Augmenter:
    """
    Randomly applies a set of realistic artefacts. Each argument is the
    probability of applying that transform (0 disables it), except
    ``noise_snr``, ``scale_range``, ``max_shift`` and ``stretch_range``
    which also set the parameter ranges.

    Parameters
    ----------
    fs            : sampling rate — needed for powerline / wander
    noise_snr     : (low, high) SNR in dB for white noise, or None
    p_noise       : probability of adding noise
    powerline     : probability of mains hum (50 or 60 Hz at random)
    wander        : probability of baseline wander
    spikes        : probability of impulse spikes
    dropout       : probability of a flat-line dropout
    scale_range   : (low, high) gain factor, or None
    max_shift     : max circular shift as a fraction of the length, or 0
    stretch_range : (low, high) time-stretch factor, or None
    invert        : probability of flipping polarity
    seed          : random seed for reproducibility
    """

    def __init__(
        self,
        fs: Optional[float] = None,
        noise_snr: Optional[tuple] = (15, 35),
        p_noise: float = 0.5,
        powerline: float = 0.0,
        wander: float = 0.0,
        spikes: float = 0.0,
        dropout: float = 0.0,
        scale_range: Optional[tuple] = (0.8, 1.2),
        max_shift: float = 0.0,
        stretch_range: Optional[tuple] = None,
        invert: float = 0.0,
        seed: Optional[int] = None,
    ):
        if (powerline or wander) and not fs:
            raise ValueError("fs is required for powerline / wander augmentation")
        self.fs = fs
        self.noise_snr, self.p_noise = noise_snr, p_noise
        self.powerline, self.wander, self.spikes = powerline, wander, spikes
        self.dropout, self.invert = dropout, invert
        self.scale_range, self.max_shift, self.stretch_range = scale_range, max_shift, stretch_range
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Augment one example (1D, or (channels, time))."""
        r = self.rng
        y = np.asarray(x, dtype=float)
        if self.stretch_range and r.random() < 0.5:
            y = time_stretch(y, r.uniform(*self.stretch_range))
        if self.max_shift:
            y = time_shift(y, int(r.uniform(-self.max_shift, self.max_shift) * y.shape[-1]))
        if self.scale_range:
            y = scale(y, r.uniform(*self.scale_range))
        if self.invert and r.random() < self.invert:
            y = invert(y)
        if self.wander and r.random() < self.wander:
            y = add_baseline_wander(y, self.fs, amplitude=r.uniform(0.2, 1.0), rng=r)
        if self.powerline and r.random() < self.powerline:
            y = add_powerline(y, self.fs, freq=r.choice([50.0, 60.0]),
                              amplitude=r.uniform(0.05, 0.3), rng=r)
        if self.noise_snr and r.random() < self.p_noise:
            y = add_noise(y, r.uniform(*self.noise_snr), rng=r)
        if self.spikes and r.random() < self.spikes:
            y = add_spikes(y, n=int(r.integers(1, 5)), rng=r)
        if self.dropout and r.random() < self.dropout:
            y = dropout(y, rng=r)
        return y

    def expand(self, X, y=None, copies: int = 1, include_original: bool = True):
        """
        Grow a dataset: ``copies`` augmented versions of every example.
        X: (n, T) or (n, C, T). Returns (X_new, y_new) — or X_new if y is None.
        """
        X = np.asarray(X, dtype=float)
        parts = [X] if include_original else []
        parts += [np.stack([self(x) for x in X]) for _ in range(copies)]
        X_new = np.concatenate(parts).astype(np.float32)
        if y is None:
            return X_new
        y = np.asarray(y)
        return X_new, np.concatenate([y] * len(parts))

    def __repr__(self) -> str:
        on = [k for k in ("powerline", "wander", "spikes", "dropout", "invert") if getattr(self, k)]
        return f"Augmenter(noise_snr={self.noise_snr}, scale={self.scale_range}, artefacts={on})"


# ═════════════════════════════════════════════════════════════════════════
# Deep-learning export
# ═════════════════════════════════════════════════════════════════════════


def make_windows(
    signals,
    fs: Optional[float] = None,
    window: float = 10.0,
    step: Optional[float] = None,
    labels: Optional[Sequence] = None,
    layout: str = "NCT",
):
    """
    Slice one or many recordings into fixed-length windows for deep learning.

    signals : a Data object, a 1D/2D array (samples[, channels]), or a list
              of those (one per recording)
    window, step : seconds (samples if fs is None)
    labels  : one label per recording → repeated for each of its windows
    layout  : 'NCT' (PyTorch Conv1d: windows, channels, time) or 'NTC'

    Returns X (float32) — and y if labels were given.
    """
    from PBstats.core.data import Data

    if isinstance(signals, (Data, np.ndarray)) and not (
            isinstance(signals, np.ndarray) and signals.dtype == object):
        signals = [signals]
        if labels is not None and np.ndim(labels) == 0:
            labels = [labels]
    if labels is not None and len(labels) != len(signals):
        raise ValueError(f"{len(labels)} labels for {len(signals)} recordings")

    xs, ys = [], []
    for k, s in enumerate(signals):
        if isinstance(s, Data):
            arr, rate = s.data, s.fs or fs
        else:
            arr, rate = np.asarray(s, dtype=float), fs
        scale_ = rate or 1.0
        win = int(round(window * scale_))
        hop = int(round((step or window) * scale_))
        arr = arr[:, None] if arr.ndim == 1 else arr
        for i in range(0, len(arr) - win + 1, hop):
            xs.append(arr[i:i + win])
            if labels is not None:
                ys.append(labels[k])
    if not xs:
        raise ValueError("No complete windows — recordings shorter than the window?")
    n_ch = {w.shape[1] for w in xs}
    if len(n_ch) > 1:
        raise ValueError(f"Recordings have different numbers of channels {sorted(n_ch)}; "
                         f"windows must share one shape")
    X = np.stack(xs).astype(np.float32)               # (N, T, C)
    if layout == "NCT":
        X = np.transpose(X, (0, 2, 1))
    elif layout != "NTC":
        raise ValueError("layout must be 'NCT' or 'NTC'")
    return (X, np.asarray(ys)) if labels is not None else X


def to_torch_dataset(X, y=None, augmenter: Optional[Augmenter] = None):
    """
    Wrap arrays as a ``torch.utils.data.Dataset``. With an ``augmenter``,
    every access returns a freshly augmented example (on-the-fly
    augmentation, so each epoch sees new variations). Requires PyTorch.
    """
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError:  # pragma: no cover
        raise ImportError("PyTorch is required: pip install torch") from None

    X = np.asarray(X, dtype=np.float32)
    y_arr = None if y is None else np.asarray(y)

    class SignalDataset(Dataset):
        def __len__(self):
            return len(X)

        def __getitem__(self, i):
            x = X[i] if augmenter is None else augmenter(X[i]).astype(np.float32)
            xt = torch.from_numpy(np.ascontiguousarray(x))
            if y_arr is None:
                return xt
            return xt, torch.as_tensor(y_arr[i])

    return SignalDataset()
