"""
ECG-specific algorithms: R-peak detection and beat-detection scoring.

``pan_tompkins`` follows Pan & Tompkins, "A Real-Time QRS Detection
Algorithm", IEEE TBME 32(3), 1985 — band-pass (5–15 Hz), derivative,
squaring, 150 ms moving-window integration, adaptive signal/noise
thresholds, search-back for missed beats and T-wave discrimination.
It runs offline with zero-phase filtering, so detections are not delayed.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


def simple_r_peaks(x, fs, distance=None, height=None) -> np.ndarray:
    """Threshold detector: find_peaks with a 0.3 s refractory period and a
    height of 50% of the 99th percentile. Fast, but assumes upright R-waves
    and a clean baseline."""
    x = np.asarray(x, dtype=float)
    distance = distance or max(int(0.3 * fs), 1)
    height = height if height is not None else 0.5 * np.percentile(x, 99)
    return find_peaks(x, distance=distance, height=height)[0]


def pan_tompkins(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Pan-Tompkins QRS detector. Returns R-peak sample indices.

    Works on either polarity (the R-peak is placed at the largest absolute
    deflection of the 5–15 Hz band-passed QRS) and does not require the
    signal to be pre-filtered. Needs fs >= 50 Hz.
    """
    x = np.asarray(x, dtype=float)
    if fs < 50:
        raise ValueError("Pan-Tompkins needs fs >= 50 Hz")
    if np.isnan(x).any():
        raise ValueError("Signal contains NaN — call remove_missing() first")
    n = len(x)
    if n < int(2 * fs):
        return np.array([], dtype=int)

    # 1–4: band-pass → derivative → square → moving-window integration
    bp = sosfiltfilt(butter(2, [5, 15], btype="bandpass", fs=fs, output="sos"), x)
    der = np.gradient(bp)
    win = max(int(round(0.150 * fs)), 1)
    mwi = np.convolve(der ** 2, np.ones(win) / win, mode="same")   # centred: no delay

    refractory = int(round(0.200 * fs))
    t_wave_window = int(round(0.360 * fs))
    candidates, _ = find_peaks(mwi, distance=refractory)
    if len(candidates) == 0:
        return np.array([], dtype=int)

    # Learning phase: initialise thresholds on the first 2 s
    first = mwi[: int(2 * fs)]
    spki = 0.25 * first.max()
    npki = 0.5 * first.mean()

    def slope(i: int) -> float:
        return float(np.max(np.abs(der[max(i - win, 0): i + 1])))

    qrs: list[int] = []
    rr_recent: list[int] = []
    for c in candidates:
        pk = mwi[c]
        thr1 = npki + 0.25 * (spki - npki)
        thr2 = 0.5 * thr1

        # Search-back: an unusually long gap means a beat was probably missed;
        # accept the strongest candidate in the gap above the lower threshold.
        if qrs and len(rr_recent) >= 2:
            rr_avg = float(np.mean(rr_recent[-8:]))
            if c - qrs[-1] > 1.66 * rr_avg:
                gap = candidates[(candidates > qrs[-1] + refractory) & (candidates < c - refractory)]
                gap = gap[mwi[gap] > thr2]
                if len(gap):
                    best = int(gap[np.argmax(mwi[gap])])
                    rr_recent.append(best - qrs[-1])
                    qrs.append(best)
                    spki = 0.25 * mwi[best] + 0.75 * spki

        if pk > thr1:
            # T-wave discrimination: a peak soon after a QRS with less than
            # half its slope is the T-wave, not a new beat.
            if qrs and c - qrs[-1] < t_wave_window and slope(c) < 0.5 * slope(qrs[-1]):
                npki = 0.125 * pk + 0.875 * npki
                continue
            if qrs:
                rr_recent.append(c - qrs[-1])
            qrs.append(int(c))
            spki = 0.125 * pk + 0.875 * spki
        else:
            npki = 0.125 * pk + 0.875 * npki

    # Locate the R-peak: largest |band-passed| deflection near each MWI peak.
    half = int(round(0.100 * fs))
    r = []
    for c in qrs:
        lo, hi = max(c - half, 0), min(c + half + 1, n)
        r.append(lo + int(np.argmax(np.abs(bp[lo:hi]))))
    return np.unique(np.asarray(r, dtype=int))


def detect_r_peaks(
    x, fs, method: str = "pantompkins", distance=None, height=None,
) -> np.ndarray:
    """R-peak indices. method: 'pantompkins' (default, robust) or 'simple'."""
    if method == "pantompkins":
        return pan_tompkins(x, fs)
    if method == "simple":
        return simple_r_peaks(x, fs, distance, height)
    raise ValueError("method must be 'pantompkins' or 'simple'")


def match_beats(reference, detected, fs: float, tolerance: float = 0.150) -> dict:
    """
    Score detections against reference beat positions (samples) with the
    standard ±150 ms tolerance and one-to-one matching.

    Returns TP, FN, FP, sensitivity (Se), positive predictivity (+P) and F1.
    """
    ref = np.sort(np.asarray(reference, dtype=int))
    det = np.sort(np.asarray(detected, dtype=int))
    tol = tolerance * fs
    used = np.zeros(len(det), dtype=bool)
    tp = 0
    j = 0
    for r in ref:
        while j < len(det) and det[j] < r - tol:
            j += 1
        best, best_d = -1, tol + 1
        k = j
        while k < len(det) and det[k] <= r + tol:
            if not used[k] and abs(det[k] - r) < best_d:
                best, best_d = k, abs(det[k] - r)
            k += 1
        if best >= 0:
            used[best] = True
            tp += 1
    fn, fp = len(ref) - tp, len(det) - tp
    se = tp / (tp + fn) if tp + fn else float("nan")
    ppv = tp / (tp + fp) if tp + fp else 0.0          # no detections → +P = 0
    f1 = 2 * se * ppv / (se + ppv) if (se + ppv) > 0 else 0.0
    return {"TP": tp, "FN": fn, "FP": fp, "Se": se, "PPV": ppv, "F1": f1}
