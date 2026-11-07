"""
Epoching: cut fixed-length segments around events and average them.

    ep = d.epochs("stimulus", tmin=-0.2, tmax=0.8, baseline=(None, 0))
    erp = ep.average()          # a Data object — plot, describe, featurise it
    ep["N"].average()           # ECG: average normal beat from annotations

The core workflow of EEG event-related potentials (ERPs), and equally useful
for averaged heartbeats, evoked EMG responses or machine-cycle analysis.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from PBstats.core.data import Data


class Epochs:
    """
    Stack of equal-length segments cut around events.

    Attributes
    ----------
    data   : (n_epochs, n_times) or (n_epochs, n_times, n_channels)
    times  : time of each sample relative to the event (s)
    labels : event label per epoch
    onsets : event time per epoch in the source recording (s)
    """

    def __init__(self, data, times, fs, labels, onsets, channel_names=None, label="",
                 n_dropped=0):
        self.data = np.asarray(data, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.fs = fs
        self.labels = np.asarray(labels, dtype=object)
        self.onsets = np.asarray(onsets, dtype=float)
        self.channel_names = channel_names
        self.label = label
        self.n_dropped = n_dropped

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        kinds = pd.Series(self.labels).value_counts().to_dict()
        return (f"Epochs(n={len(self)}, times=[{self.times[0]:.3f}, {self.times[-1]:.3f}] s, "
                f"labels={kinds}, dropped={self.n_dropped})")

    def __getitem__(self, key) -> "Epochs":
        """Select by label (``ep["N"]``, ``ep[["N", "V"]]``), mask or indices."""
        if isinstance(key, str):
            key = self.labels == key
        elif isinstance(key, (list, tuple)) and key and isinstance(key[0], str):
            key = np.isin(self.labels, key)
        return Epochs(self.data[key], self.times, self.fs, self.labels[key], self.onsets[key],
                      self.channel_names, self.label)

    def average(self) -> "Data":
        """Mean across epochs → Data (the ERP / average beat). Noise not
        time-locked to the event shrinks by about √n_epochs."""
        return self._summary(np.mean, "average")

    def median(self) -> "Data":
        """Median across epochs — robust to a few artefact-laden epochs."""
        return self._summary(np.median, "median")

    def std(self) -> "Data":
        return self._summary(np.std, "std")

    def _summary(self, fn, name) -> "Data":
        from PBstats.core.data import Data
        if len(self) == 0:
            raise ValueError("No epochs to summarise")
        out = Data(fn(self.data, axis=0), fs=self.fs, label=f"{self.label} {name} (n={len(self)})",
                   channel_names=self.channel_names if self.data.ndim == 3 else None)
        out.meta["epoch_times"] = self.times
        return out

    def to_numpy(self, layout: str = "NCT") -> np.ndarray:
        """Array for deep learning. 'NCT' = (epochs, channels, times) —
        PyTorch Conv1d layout; 'NTC' = (epochs, times, channels)."""
        x = self.data if self.data.ndim == 3 else self.data[:, :, None]
        if layout == "NTC":
            return x.astype(np.float32)
        if layout == "NCT":
            return np.transpose(x, (0, 2, 1)).astype(np.float32)
        raise ValueError("layout must be 'NCT' or 'NTC'")

    def plot(self, save_path: Optional[str] = None, max_traces: int = 50):
        """Overlay of individual epochs (thin) and their mean ± SD (bold)."""
        from PBstats.visualization.plots import plot_epochs
        return plot_epochs(self, save_path=save_path, max_traces=max_traces)


def make_epochs(
    d: "Data",
    events: Union[None, str, Sequence, np.ndarray, pd.DataFrame] = None,
    tmin: float = -0.2,
    tmax: float = 0.5,
    baseline: Optional[tuple] = (None, 0.0),
    reject: Optional[float] = None,
) -> Epochs:
    if not d.fs:
        raise ValueError("epochs() requires fs")
    if tmax <= tmin:
        raise ValueError("tmax must be greater than tmin")

    ann = d.annotations
    if events is None or isinstance(events, str) or (
            isinstance(events, (list, tuple)) and events and isinstance(events[0], str)):
        if ann is None or len(ann) == 0:
            raise ValueError("No annotations on this Data; pass event times explicitly")
        sel = ann if events is None else ann[ann["label"].isin(
            [events] if isinstance(events, str) else list(events))]
        onsets, labels = sel["onset"].to_numpy(float), sel["label"].astype(str).to_numpy()
    elif isinstance(events, pd.DataFrame):
        onsets = events["onset"].to_numpy(float)
        labels = events["label"].astype(str).to_numpy() if "label" in events else np.full(len(onsets), "event")
    else:
        onsets = np.asarray(events, dtype=float)
        labels = np.full(len(onsets), "event", dtype=object)

    i0 = int(round(tmin * d.fs))
    i1 = int(round(tmax * d.fs))
    times = np.arange(i0, i1 + 1) / d.fs
    x = d.data
    segs, keep_lab, keep_on = [], [], []
    dropped = 0
    for t, lab in zip(onsets, labels):
        c = int(round(t * d.fs))
        if c + i0 < 0 or c + i1 + 1 > len(x):
            dropped += 1
            continue
        seg = x[c + i0: c + i1 + 1].astype(float).copy()
        if baseline is not None:
            b0 = times[0] if baseline[0] is None else baseline[0]
            b1 = times[-1] if baseline[1] is None else baseline[1]
            bmask = (times >= b0) & (times <= b1)
            if bmask.any():
                seg = seg - seg[bmask].mean(axis=0)
        if reject is not None and np.max(np.ptp(seg, axis=0)) > reject:
            dropped += 1
            continue
        segs.append(seg)
        keep_lab.append(lab)
        keep_on.append(t)

    shape = (0, len(times)) + (x.shape[1:] if x.ndim == 2 else ())
    data = np.array(segs) if segs else np.empty(shape)
    return Epochs(data, times, d.fs, keep_lab, keep_on,
                  d.channel_names if x.ndim == 2 else None, d.label, dropped)


class EpochsMixin:

    def epochs(
        self,
        events=None,
        tmin: float = -0.2,
        tmax: float = 0.5,
        baseline: Optional[tuple] = (None, 0.0),
        reject: Optional[float] = None,
    ) -> Epochs:
        """
        Cut segments [tmin, tmax] s around events.

        events   : event times in seconds; a DataFrame with ``onset`` (and
                   ``label``); an annotation label / list of labels (e.g. "N"
                   for normal beats in PhysioNet records); or None for every
                   annotation on this Data.
        baseline : (start, end) s relative to the event; the mean over this
                   interval is subtracted per epoch. None to skip.
                   Default (None, 0) = everything before the event.
        reject   : drop epochs whose peak-to-peak amplitude exceeds this
                   (artefact rejection)

        Epochs too close to the recording edges are dropped (counted in
        ``.n_dropped``).
        """
        return make_epochs(self, events, tmin, tmax, baseline, reject)
