"""
Pure feature functions: 1D array in → dict (or float) out.

These are independent of the ``Data`` class so they can be used directly,
in sliding windows, or inside the scikit-learn transformer.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats as sp_stats
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, peak_prominences, peak_widths, welch

EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}

NAN = float("nan")


# ═════════════════════════════════════════════════════════════════════════
# Nonlinear / complexity measures
# ═════════════════════════════════════════════════════════════════════════

def sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """
    Sample entropy (Richman & Moorman 2000): −ln(A/B), where B and A count
    template pairs of length m and m+1 within Chebyshev distance r
    (self-matches excluded). Low = regular / predictable, high = irregular.

    Uses a KD-tree, so it is O(n log n) rather than the naive O(n²) loop.
    Returns NaN when undefined (no matches — signal too short or r too small).

    r defaults to 0.2 × std(x), the conventional choice.
    """
    from sklearn.neighbors import KDTree

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if r is None:
        r = 0.2 * np.std(x)
    if n <= m + 2 or r <= 0:
        return NAN

    def _pairs(length: int) -> int:
        # Same number of templates (n - m) for both lengths, per the definition.
        emb = sliding_window_view(x, length)[: n - m]
        tree = KDTree(emb, metric="chebyshev")
        counts = tree.query_radius(emb, r, count_only=True)
        return int(counts.sum() - len(emb))  # remove self-matches

    b = _pairs(m)
    a = _pairs(m + 1)
    if a == 0 or b == 0:
        return NAN
    return float(-np.log(a / b))


def permutation_entropy(
    x: np.ndarray, order: int = 3, delay: int = 1, normalize: bool = True,
) -> float:
    """
    Permutation entropy (Bandt & Pompe 2002): Shannon entropy of ordinal
    patterns. Robust to noise and amplitude scaling. With ``normalize`` the
    result is in [0, 1] (1 = all patterns equally likely, i.e. white noise).
    """
    x = np.asarray(x, dtype=float)
    span = (order - 1) * delay + 1
    if len(x) < span + 1:
        return NAN
    emb = sliding_window_view(x, span)[:, ::delay]
    patterns = np.argsort(emb, axis=1, kind="stable")
    # Encode each permutation as one integer for fast counting.
    codes = patterns @ (order ** np.arange(order))
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    h = max(float(-np.sum(p * np.log2(p))), 0.0)   # avoid -0.0
    return h / math.log2(math.factorial(order)) if normalize else h


def higuchi_fd(x: np.ndarray, kmax: int = 10) -> float:
    """
    Higuchi fractal dimension — waveform complexity in [1, 2].
    ~1 for smooth signals (sine), ~1.5 for Brownian motion, ~2 for white noise.
    Widely used for EEG (sleep staging, seizure detection).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    kmax = int(min(kmax, n // 4))
    if kmax < 2:
        return NAN
    ks, lengths = [], []
    for k in range(1, kmax + 1):
        lm = []
        for m in range(k):
            sub = x[m::k]
            if len(sub) < 2:
                continue
            norm = (n - 1) / ((len(sub) - 1) * k)
            lm.append(np.abs(np.diff(sub)).sum() * norm / k)
        if lm:
            ks.append(k)
            lengths.append(np.mean(lm))
    lengths = np.asarray(lengths)
    if np.any(lengths <= 0):
        return NAN
    slope = np.polyfit(np.log(1.0 / np.asarray(ks)), np.log(lengths), 1)[0]
    return float(slope)


def dfa_alpha(x: np.ndarray, order: int = 1, n_scales: int = 12) -> float:
    """
    Detrended Fluctuation Analysis scaling exponent α — long-range temporal
    correlation. 0.5 = uncorrelated (white noise), 1.0 = 1/f (pink) noise,
    1.5 = Brownian motion. Classic HRV and EEG biomarker; robust to
    non-stationarity.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 32:
        return NAN
    y = np.cumsum(x - x.mean())
    scales = np.unique(np.floor(np.geomspace(max(order + 3, 8), n // 4, n_scales)).astype(int))
    scales = scales[scales > order + 2]
    if len(scales) < 3:
        return NAN

    fluct = []
    for s in scales:
        n_seg = n // s
        segs = y[: n_seg * s].reshape(n_seg, s)
        t = np.arange(s, dtype=float)
        coef = np.polyfit(t, segs.T, order)                 # (order+1, n_seg)
        trend = np.vander(t, order + 1) @ coef               # (s, n_seg)
        fluct.append(np.sqrt(np.mean((segs.T - trend) ** 2)))
    fluct = np.asarray(fluct)
    if np.any(fluct <= 0):
        return NAN
    return float(np.polyfit(np.log(scales), np.log(fluct), 1)[0])


def hjorth(x: np.ndarray) -> tuple[float, float, float]:
    """Hjorth activity, mobility, complexity."""
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    ddx = np.diff(dx)
    vx, vdx, vddx = np.var(x), np.var(dx), np.var(ddx)
    mobility = np.sqrt(vdx / vx) if vx > 0 else 0.0
    mob_d = np.sqrt(vddx / vdx) if vdx > 0 else 0.0
    complexity = mob_d / mobility if mobility > 0 else 0.0
    return float(vx), float(mobility), float(complexity)


# ═════════════════════════════════════════════════════════════════════════
# Feature groups
# ═════════════════════════════════════════════════════════════════════════

def time_domain(x: np.ndarray) -> dict:
    """Amplitude, shape and waveform features (prefix ``td_``)."""
    x = np.asarray(x, dtype=float)
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    activity, mobility, complexity = hjorth(x)
    centred = np.signbit(x - x.mean())
    with warnings.catch_warnings():  # constant input → NaN skew, that's fine
        warnings.simplefilter("ignore", RuntimeWarning)
        skew = float(sp_stats.skew(x))
        kurt = float(sp_stats.kurtosis(x))

    return {
        "td_mean": float(np.mean(x)),
        "td_std": std,
        "td_variance": std ** 2,
        "td_skewness": skew,
        "td_kurtosis": kurt,
        "td_rms": rms,
        "td_peak_to_peak": float(np.ptp(x)),
        "td_crest_factor": peak / rms if rms > 0 else 0.0,
        "td_zero_crossing_rate": float(np.count_nonzero(centred[1:] != centred[:-1])
                                       / max(len(x) - 1, 1)),
        "td_mean_abs_value": float(np.mean(np.abs(x))),
        "td_waveform_length": float(np.sum(np.abs(np.diff(x)))),
        "td_hjorth_activity": activity,
        "td_hjorth_mobility": mobility,
        "td_hjorth_complexity": complexity,
    }


def frequency_domain(x: np.ndarray, fs: float, nperseg: int | None = None) -> dict:
    """
    Spectral features from a Welch PSD (prefix ``fd_``). Welch averaging gives
    far more stable features than a single periodogram — important for ML,
    where feature noise directly becomes model noise.

    Band powers are *relative* (fraction of total power); bands above the
    Nyquist frequency are reported as 0.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    nperseg = int(min(nperseg or max(int(2 * fs), 8), len(x)))
    freqs, psd = welch(x, fs=fs, nperseg=nperseg)

    total = psd.sum()
    if total <= 0:
        p = np.full_like(psd, 1.0 / len(psd))
    else:
        p = psd / total

    centroid = float(np.sum(freqs * p))
    spread = float(np.sqrt(np.sum((freqs - centroid) ** 2 * p)))
    skew = float(np.sum((freqs - centroid) ** 3 * p) / spread ** 3) if spread > 0 else 0.0
    kurt = float(np.sum((freqs - centroid) ** 4 * p) / spread ** 4) if spread > 0 else 0.0
    flatness = float(np.exp(np.mean(np.log(psd + 1e-300))) / (np.mean(psd) + 1e-300))
    entropy = float(-np.sum(p[p > 0] * np.log2(p[p > 0])) / np.log2(len(p)))

    cum = np.cumsum(p)

    def edge(q: float) -> float:
        return float(freqs[min(np.searchsorted(cum, q), len(freqs) - 1)])

    total_area = trapezoid(psd, freqs) if total > 0 else 0.0

    def rel_band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        if total_area <= 0 or mask.sum() == 0:
            return 0.0
        if mask.sum() == 1:
            return float(psd[mask].sum() * (freqs[1] - freqs[0]) / total_area)
        return float(trapezoid(psd[mask], freqs[mask]) / total_area)

    feats = {
        "fd_spectral_centroid": centroid,
        "fd_spectral_spread": spread,
        "fd_spectral_skewness": skew,
        "fd_spectral_kurtosis": kurt,
        "fd_spectral_flatness": flatness,
        "fd_spectral_rolloff": edge(0.85),
        "fd_spectral_entropy": entropy,
        "fd_dominant_frequency": float(freqs[np.argmax(psd)]),
        "fd_sef_90hz": edge(0.90),
        "fd_sef_95hz": edge(0.95),
    }
    for name, (lo, hi) in EEG_BANDS.items():
        feats[f"fd_band_{name}"] = rel_band(lo, hi)
    return feats


def nonlinear(x: np.ndarray) -> dict:
    """Complexity / fractal features (prefix ``nl_``)."""
    return {
        "nl_sample_entropy": sample_entropy(x),
        "nl_permutation_entropy": permutation_entropy(x),
        "nl_higuchi_fd": higuchi_fd(x),
        "nl_dfa_alpha": dfa_alpha(x),
    }


def peaks(
    x: np.ndarray, fs: float | None, min_distance: int | None = None,
    min_height: float | None = None,
) -> dict:
    """Event / peak statistics (prefix ``pk_``)."""
    x = np.asarray(x, dtype=float)
    dist = min_distance or max(int((fs or 100) * 0.1), 1)
    height = min_height if min_height is not None else float(np.mean(x) + 0.5 * np.std(x))
    idx, props = find_peaks(x, distance=dist, height=height)
    duration = len(x) / (fs or 1.0)

    if len(idx) == 0:
        return {"pk_count": 0.0, "pk_rate_hz": 0.0, "pk_mean_height": 0.0,
                "pk_std_height": 0.0, "pk_mean_prominence": 0.0,
                "pk_mean_width": 0.0, "pk_regularity": 0.0}

    proms = peak_prominences(x, idx)[0]
    widths = peak_widths(x, idx, rel_height=0.5)[0] / (fs or 1.0)
    ipi = np.diff(idx)
    return {
        "pk_count": float(len(idx)),
        "pk_rate_hz": float(len(idx) / duration),
        "pk_mean_height": float(x[idx].mean()),
        "pk_std_height": float(x[idx].std()),
        "pk_mean_prominence": float(proms.mean()),
        "pk_mean_width": float(widths.mean()),   # seconds if fs set, else samples
        "pk_regularity": float(ipi.std() / ipi.mean()) if len(ipi) > 1 else 0.0,
    }


def detect_r_peaks(
    x: np.ndarray, fs: float, distance: int | None = None, height: float | None = None,
) -> np.ndarray:
    """Simple R-peak detector: find_peaks with refractory period (default 0.3 s
    → up to 200 bpm) and height threshold (default 50% of the 99th percentile,
    which is robust to a single artefact spike)."""
    x = np.asarray(x, dtype=float)
    distance = distance or max(int(0.3 * fs), 1)
    height = height if height is not None else 0.5 * np.percentile(x, 99)
    return find_peaks(x, distance=distance, height=height)[0]


HRV_KEYS = (
    "hrv_beats", "hrv_mean_rr_ms", "hrv_sdnn_ms", "hrv_rmssd_ms", "hrv_pnn50_pct",
    "hrv_mean_hr_bpm", "hrv_sdhr_bpm", "hrv_rr_range_ms", "hrv_tri_index",
    "hrv_sd1_ms", "hrv_sd2_ms", "hrv_sd1_sd2",
    "hrv_lf_power", "hrv_hf_power", "hrv_lf_hf_ratio", "hrv_artifact_pct",
)


def clean_rr(rr_ms: np.ndarray, rel: float | None = 0.3, window: int = 11) -> np.ndarray:
    """
    NN-interval filtering — boolean mask of R-R intervals to keep.

    Rejects physiologically implausible intervals (< 250 ms or > 2500 ms,
    i.e. outside 24–240 bpm) and, if ``rel`` is set, intervals deviating more
    than ``rel`` (fraction) from the median of the surrounding ``window``
    beats — the signature of missed/extra R-peak detections and ectopic
    beats. A single such artefact can double SDNN, so this matters.
    """
    from scipy.ndimage import median_filter

    rr = np.asarray(rr_ms, dtype=float)
    ok = (rr >= 250) & (rr <= 2500)
    if rel and len(rr) >= 5:
        w = min(window, len(rr) - (1 - len(rr) % 2))   # odd, <= len
        # mirror padding: "nearest" would replicate an outlier sitting at the
        # edge until it became its own local median
        local = median_filter(rr, size=max(w, 3), mode="mirror")
        ok &= np.abs(rr - local) <= rel * local
    return ok


def hrv(rr_ms: np.ndarray, rr_filter: float | None = 0.3) -> dict:
    """
    Heart-rate-variability metrics from R-R intervals in ms (prefix ``hrv_``).

    Intervals are first cleaned with :func:`clean_rr` (``rr_filter`` = max
    relative deviation from the local median; None disables it). The share
    rejected is reported as ``hrv_artifact_pct`` — itself a useful feature
    (signal quality / rhythm irregularity).

    Time domain: mean RR, SDNN, RMSSD, pNN50, mean/SD heart rate, range,
    triangular index. Poincaré: SD1 (short-term, vagal), SD2 (long-term),
    SD1/SD2. Frequency domain (needs ≥ 20 beats): LF (0.04–0.15 Hz),
    HF (0.15–0.4 Hz), LF/HF — from a 4 Hz resampled tachogram.
    """
    rr_raw = np.asarray(rr_ms, dtype=float)
    # Always return the full key set (NaN when not computable) so feature
    # matrices keep a fixed width across recordings.
    out = {k: NAN for k in HRV_KEYS}
    out["hrv_beats"] = float(len(rr_raw) + 1) if len(rr_raw) else 0.0
    if len(rr_raw) < 3:
        return out

    keep = clean_rr(rr_raw, rr_filter)
    out["hrv_artifact_pct"] = float(100 * (1 - keep.mean()))
    beat_times = np.cumsum(rr_raw) / 1000.0      # true timing, before rejection
    rr = rr_raw[keep]
    if len(rr) < 3:
        return out

    drr = np.diff(rr)
    hr = 60000.0 / rr
    hist, _ = np.histogram(rr, bins=np.arange(rr.min(), rr.max() + 7.8125, 7.8125))
    sdnn = float(np.std(rr, ddof=1))
    sd1 = float(np.std(drr, ddof=1) / np.sqrt(2))
    sd2 = float(np.sqrt(max(2 * sdnn ** 2 - sd1 ** 2, 0.0)))
    out.update({
        "hrv_mean_rr_ms": float(rr.mean()),
        "hrv_sdnn_ms": sdnn,
        "hrv_rmssd_ms": float(np.sqrt(np.mean(drr ** 2))),
        "hrv_pnn50_pct": float(np.mean(np.abs(drr) > 50) * 100),
        "hrv_mean_hr_bpm": float(hr.mean()),
        "hrv_sdhr_bpm": float(np.std(hr, ddof=1)),
        "hrv_rr_range_ms": float(np.ptp(rr)),
        # standard 1/128 s (7.8125 ms) bins
        "hrv_tri_index": float(len(rr) / hist.max()) if hist.size and hist.max() > 0 else NAN,
        "hrv_sd1_ms": sd1,
        "hrv_sd2_ms": sd2,
        "hrv_sd1_sd2": sd1 / sd2 if sd2 > 0 else NAN,
    })

    if len(rr) >= 20:
        t = beat_times[keep]
        t_uniform = np.arange(t[0], t[-1], 0.25)            # 4 Hz
        if len(t_uniform) >= 16:
            tach = np.interp(t_uniform, t, rr)
            f, pxx = welch(tach - tach.mean(), fs=4.0, nperseg=min(256, len(tach)))
            lf_m = (f >= 0.04) & (f < 0.15)
            hf_m = (f >= 0.15) & (f < 0.40)
            lf = float(trapezoid(pxx[lf_m], f[lf_m])) if lf_m.sum() > 1 else NAN
            hf = float(trapezoid(pxx[hf_m], f[hf_m])) if hf_m.sum() > 1 else NAN
            out.update({
                "hrv_lf_power": lf,
                "hrv_hf_power": hf,
                "hrv_lf_hf_ratio": lf / hf if hf and hf > 0 else NAN,
            })
    return out


# ═════════════════════════════════════════════════════════════════════════
# One-call extraction
# ═════════════════════════════════════════════════════════════════════════

CATEGORIES = ("time", "freq", "nonlinear", "peaks", "hrv")
DEFAULT_CATEGORIES = ("time", "freq", "nonlinear", "peaks")


def extract(x: np.ndarray, fs: float | None = None, include=None) -> dict:
    """
    Extract a flat feature dict from a 1D signal.

    include : iterable of categories from
              'time', 'freq', 'nonlinear', 'peaks', 'hrv'
              (default: all except 'hrv', which only makes sense for ECG).
    """
    include = tuple(include) if include is not None else DEFAULT_CATEGORIES
    unknown = set(include) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown feature categories {sorted(unknown)}. Choose from {CATEGORIES}")
    x = np.asarray(x, dtype=float).flatten()
    if np.isnan(x).any():
        raise ValueError("Signal contains NaN — call remove_missing() before extracting features.")

    feats: dict = {}
    if "time" in include:
        feats.update(time_domain(x))
    if "freq" in include:
        if not fs:
            raise ValueError("fs is required for frequency features.")
        feats.update(frequency_domain(x, fs))
    if "nonlinear" in include:
        feats.update(nonlinear(x))
    if "peaks" in include:
        feats.update(peaks(x, fs))
    if "hrv" in include:
        if not fs:
            raise ValueError("fs is required for HRV features.")
        r = detect_r_peaks(x, fs)
        feats.update(hrv(np.diff(r) / fs * 1000.0))
    return feats
