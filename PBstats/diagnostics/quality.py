"""
Signal diagnosis — PBstats' "signal doctor".

``Data.diagnose()`` inspects a raw recording for the problems that silently
ruin downstream analysis (gaps, sensor dropout, clipping, impulse spikes,
baseline drift, power-line interference, broadband noise, non-stationarity),
grades it 0–100, explains each finding, and *prescribes* an ordered cleaning
``Pipeline``. ``Data.auto_clean()`` applies that prescription.

Every check is a cheap heuristic with documented thresholds; the report is
meant to make problems visible and give a sensible default fix, not to
replace domain judgement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import welch

from PBstats.core.steps import step
from PBstats.core.utils import interp_nans, require_1d, safe_print
from PBstats.preprocessing.outliers import detect_outliers

_PENALTY = {"critical": 35, "warning": 12, "info": 3}
_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

# Canonical order in which fixes are applied: gaps first (filters can't handle
# NaN), spikes before filtering (IIR filters smear impulses into ringing),
# then drift, then line noise, then offset.
_FIX_ORDER = ["remove_missing", "remove_outliers", "highpass", "detrend", "notch", "remove_dc"]

# Diagnosis thresholds (documented, tunable via diagnose() kwargs where it matters)
SPIKE_THRESHOLD = 10.0     # × max(noise, local slope) away from a 5-sample rolling median
MAINS_DB = 10.0            # line peak this far above local background → interference
MAINS_MIN_SHARE = 0.002    # ...and carrying at least this share of total power
DRIFT_FRACTION = 0.2       # share of power below the drift cutoff
FLAT_FRACTION = 0.01       # share of samples in flat runs
DIGITAL_LEVELS = 16        # ≤ this many distinct values → treat as a digital signal


@dataclass
class Issue:
    code: str                                  # machine-readable id, e.g. "mains"
    severity: str                              # 'critical' | 'warning' | 'info'
    message: str                               # human explanation
    value: float = float("nan")                # the measured quantity
    fix: Optional[tuple[str, dict]] = None     # (Data method, kwargs) or None


@dataclass
class QualityReport:
    label: str
    n_samples: int
    fs: Optional[float]
    issues: list[Issue] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    masks: dict = field(default_factory=dict, repr=False)   # boolean arrays per issue
    signal: np.ndarray = field(default=None, repr=False)

    # ── scoring ───────────────────────────────────────────────────────────

    @property
    def score(self) -> int:
        """0–100 quality score (100 = no issues found)."""
        return int(max(0, 100 - sum(_PENALTY[i.severity] for i in self.issues)))

    @property
    def grade(self) -> str:
        s = self.score
        return "A" if s >= 90 else "B" if s >= 75 else "C" if s >= 60 else "D" if s >= 40 else "F"

    @property
    def ok(self) -> bool:
        """True when there are no warnings or critical issues."""
        return not any(i.severity in ("critical", "warning") for i in self.issues)

    def has(self, code: str) -> bool:
        return any(i.code == code for i in self.issues)

    # ── prescription ──────────────────────────────────────────────────────

    @property
    def fixes(self) -> list[tuple[str, dict]]:
        """Recommended (method, kwargs) steps in a safe application order."""
        fixes = [i.fix for i in self.issues if i.fix is not None]
        names = [f[0] for f in fixes]
        # A highpass/detrend already removes the DC offset.
        if ("highpass" in names or "detrend" in names) and "remove_dc" in names:
            fixes = [f for f in fixes if f[0] != "remove_dc"]
        return sorted(fixes, key=lambda f: _FIX_ORDER.index(f[0]))

    @property
    def pipeline(self):
        """The prescription as a reusable, saveable ``Pipeline``."""
        from PBstats.core.pipeline import Pipeline
        pipe = Pipeline(f"auto_clean_{self.label}" if self.label else "auto_clean")
        for name, kwargs in self.fixes:
            pipe.add(name, **kwargs)
        return pipe

    # ── output ────────────────────────────────────────────────────────────

    def summary(self) -> None:
        bar = "█" * (self.score // 5) + "░" * (20 - self.score // 5)
        dur = f", {self.n_samples / self.fs:.1f} s @ {self.fs:g} Hz" if self.fs else ""
        safe_print(f"\nSignal diagnosis — '{self.label}'  ({self.n_samples} samples{dur})")
        safe_print("─" * 64)
        safe_print(f"  Quality  {bar}  {self.score}/100  grade {self.grade}")
        safe_print("─" * 64)
        if not self.issues:
            safe_print("  No issues detected.")
        icon = {"critical": "✖", "warning": "▲", "info": "•"}
        for i in sorted(self.issues, key=lambda i: _SEVERITY_ORDER[i.severity]):
            safe_print(f"  {icon[i.severity]} [{i.severity:8s}] {i.message}")
        if self.fixes:
            safe_print("─" * 64)
            safe_print("  Prescription (apply with .auto_clean() or report.pipeline):")
            for k, (name, kw) in enumerate(self.fixes, 1):
                args = ", ".join(f"{a}={v!r}" for a, v in kw.items())
                safe_print(f"    {k}. {name}({args})")
        safe_print("─" * 64)

    def to_dict(self) -> dict:
        return {
            "label": self.label, "score": self.score, "grade": self.grade,
            "issues": [{"code": i.code, "severity": i.severity, "message": i.message,
                        "value": i.value} for i in self.issues],
            "metrics": dict(self.metrics),
            "prescription": [{"step": n, "kwargs": kw} for n, kw in self.fixes],
        }

    def plot(self, save_path: str | None = None):
        from PBstats.visualization.plots import plot_quality
        return plot_quality(self, save_path=save_path)

    def __repr__(self) -> str:
        codes = ", ".join(i.code for i in self.issues) or "none"
        return f"QualityReport('{self.label}', score={self.score}, grade={self.grade}, issues=[{codes}])"


# ═════════════════════════════════════════════════════════════════════════
# Individual checks
# ═════════════════════════════════════════════════════════════════════════

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) of each run of True values."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _robust_std(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def diagnose_signal(
    x: np.ndarray,
    fs: float | None = None,
    label: str = "",
    mains: float | None = None,
    drift_cutoff: float = 0.5,
    signal_type: str | None = None,
) -> QualityReport:
    """Run every check on a 1D signal. See ``QualityMixin.diagnose``."""
    if signal_type not in (None, "ecg"):
        raise ValueError("signal_type must be None or 'ecg'")
    x = np.asarray(x, dtype=float)
    n = len(x)
    rep = QualityReport(label=label, n_samples=n, fs=fs, signal=x.copy())
    m = rep.metrics
    m["n_samples"] = n
    if fs:
        m["duration_s"] = n / fs

    def add(*args, **kwargs):
        rep.issues.append(Issue(*args, **kwargs))

    # ── 1. Missing values ─────────────────────────────────────────────────
    nan_mask = np.isnan(x)
    m["nan_fraction"] = float(nan_mask.mean()) if n else 0.0
    rep.masks["missing"] = nan_mask
    if nan_mask.all():
        add("empty", "critical", "Signal is entirely NaN — nothing to analyse.")
        return rep
    if nan_mask.any():
        frac = m["nan_fraction"]
        add("missing", "critical" if frac > 0.2 else "warning",
            f"{nan_mask.sum()} missing samples ({frac:.1%}) in "
            f"{len(_runs(nan_mask))} gap(s) — filters and FFTs fail on NaN.",
            frac, ("remove_missing", {"strategy": "interpolate"}))
        x = interp_nans(x)

    if n < 32:
        add("short", "critical", f"Only {n} samples — too short for reliable statistics or spectra.", n)
        return rep

    spread = _robust_std(x) or float(np.std(x))
    if spread == 0:
        add("constant", "critical", "Signal is constant — no information content.")
        return rep

    # Digital / few-level signals (binary triggers, square waves, step
    # counters) sit at their extremes and repeat values by design, so the
    # clipping and flatline checks below would be meaningless for them.
    m["distinct_levels"] = n_levels = int(len(np.unique(x)))
    digital = n_levels <= DIGITAL_LEVELS
    if digital:
        add("digital", "info",
            f"Signal takes only {n_levels} distinct values (digital or heavily quantized) — "
            f"clipping/flatline checks skipped.", n_levels)

    # ── 2. Clipping / saturation ──────────────────────────────────────────
    clip_mask = np.zeros(n, bool)
    for extreme in (() if digital else (x.max(), x.min())):
        at = x == extreme
        long_runs = [(a, b) for a, b in _runs(at) if b - a >= 3]
        if at.sum() >= max(3, 0.002 * n) and long_runs:
            for a, b in long_runs:
                clip_mask[a:b] = True
    m["clip_fraction"] = float(clip_mask.mean())
    rep.masks["clipping"] = clip_mask
    if clip_mask.any():
        frac = m["clip_fraction"]
        add("clipping", "critical" if frac > 0.05 else "warning",
            f"Signal saturates at its min/max for {frac:.1%} of samples — amplitude "
            f"information there is lost (check sensor gain/range). Not auto-fixable.",
            frac)

    # ── 3. Flatlines (sensor dropout) ─────────────────────────────────────
    min_run = max(5, int(0.1 * fs)) if fs else 10
    flat = np.zeros(n, bool)
    same = np.concatenate([[False], np.diff(x) == 0])
    for a, b in ([] if digital else _runs(same)):
        if b - a + 1 >= min_run:
            flat[max(a - 1, 0):b] = True
    flat &= ~clip_mask
    m["flat_fraction"] = float(flat.mean())
    rep.masks["flatline"] = flat
    if m["flat_fraction"] > FLAT_FRACTION:
        frac = m["flat_fraction"]
        add("flatline", "critical" if frac > 0.2 else "warning",
            f"{frac:.1%} of samples sit in perfectly flat segments (≥ {min_run} samples) — "
            f"likely sensor dropout or a disconnected lead. Consider cropping them out.",
            frac)

    # ── 4. Impulse spikes ─────────────────────────────────────────────────
    spikes = detect_outliers(x, "spike", threshold=SPIKE_THRESHOLD, window=5) & ~clip_mask
    m["spike_count"] = int(spikes.sum())
    rep.masks["spikes"] = spikes
    if spikes.any():
        add("spikes", "warning",
            f"{spikes.sum()} impulse spike sample(s) (> {SPIKE_THRESHOLD:g}× both noise level and "
            f"local slope) — these ring through filters and inflate variance.",
            float(spikes.sum()),
            ("remove_outliers", {"method": "spike", "threshold": SPIKE_THRESHOLD, "window": 5}))
        x = x.copy()
        idx = np.arange(n)
        x[spikes] = np.interp(idx[spikes], idx[~spikes], x[~spikes])

    # ── 5. DC offset ──────────────────────────────────────────────────────
    dc_ratio = abs(float(np.mean(x))) / (float(np.std(x)) + 1e-300)
    m["dc_ratio"] = dc_ratio
    if dc_ratio > 1.0:
        add("dc_offset", "info",
            f"Large DC offset (|mean| = {dc_ratio:.1f}× std) — dominates the 0 Hz bin of spectra.",
            dc_ratio, ("remove_dc", {}))

    # ── 6. Spectral checks (need fs) ──────────────────────────────────────
    if fs:
        nyq = fs / 2.0
        xc = x - x.mean()
        nper = int(min(n, max(8 * fs, 64)))     # ≥ 0.125 Hz resolution when possible
        f, p = welch(xc, fs=fs, nperseg=nper)
        total = p.sum() + 1e-300

        # 6a. Baseline drift / wander (needs ≥ 2 periods of the cutoff to resolve)
        if drift_cutoff:
            if drift_cutoff < nyq and n / fs >= 2.0 / drift_cutoff:
                frac = float(p[(f > 0) & (f < drift_cutoff)].sum() / total)
                m["drift_fraction"] = frac
                if frac > DRIFT_FRACTION:
                    add("drift", "warning",
                        f"{frac:.0%} of the power is below {drift_cutoff:g} Hz — baseline "
                        f"wander/drift. (If slow content is your signal, e.g. respiration, "
                        f"pass drift_cutoff=None.)",
                        frac, ("highpass", {"cutoff": drift_cutoff}))
            else:
                _trend_check(x, rep, add)   # too short to resolve: fall back to a trend fit

        # 6b. Power-line interference
        candidates = [mains] if mains else [50.0, 60.0]
        f2, p2 = welch(xc, fs=fs, nperseg=int(min(n, max(2 * fs, 64))))  # 0.5 Hz res
        total2 = p2.sum() + 1e-300
        best = None
        for f0 in candidates:
            if f0 + 3 >= nyq:
                continue
            peak_band = (f2 >= f0 - 1) & (f2 <= f0 + 1)
            bg_band = ((f2 >= f0 - 10) & (f2 <= f0 - 3)) | ((f2 >= f0 + 3) & (f2 <= min(f0 + 10, nyq)))
            if not peak_band.any() or bg_band.sum() < 3:
                continue
            ratio_db = 10 * np.log10(p2[peak_band].max() / (np.median(p2[bg_band]) + 1e-300))
            share = float(p2[peak_band].sum() / total2)
            if best is None or ratio_db > best[1]:
                best = (f0, float(ratio_db), share)
        if best:
            f0, ratio_db, share = best
            m["mains_hz"], m["mains_db"], m["mains_power_fraction"] = f0, ratio_db, share
            # Both tests matter: dB-over-background alone fires on numerically
            # noiseless signals, where the "background" is round-off.
            if ratio_db > MAINS_DB and share >= MAINS_MIN_SHARE:
                harmonics = bool(2 * f0 < nyq and _line_db(f2, p2, 2 * f0) > MAINS_DB)
                if share > 0.5:
                    add("mains", "info",
                        f"Very strong {f0:g} Hz component ({share:.0%} of power). If this is "
                        f"power-line interference rather than your signal, apply notch({f0:g}).",
                        ratio_db)
                else:
                    add("mains", "critical" if ratio_db > 30 else "warning",
                        f"Power-line interference at {f0:g} Hz ({ratio_db:.0f} dB above the "
                        f"local background{', with harmonics' if harmonics else ''}).",
                        ratio_db, ("notch", {"freq": f0, "harmonics": harmonics}))

        # 6c. Broadband high-frequency noise
        hf = float(p[f >= 0.8 * nyq].sum() / total)
        m["hf_fraction"] = hf
        if hf > 0.3:
            add("hf_noise", "info",
                f"{hf:.0%} of the power is in the top 20% of the band (> {0.8 * nyq:g} Hz) — "
                f"broadband noise or under-sampling. Consider a lowpass if your signal of "
                f"interest is lower.", hf)
    elif drift_cutoff:
        _trend_check(x, rep, add)

    # ── 7. Amplitude non-stationarity ─────────────────────────────────────
    if n >= 8 * 32:
        segs = np.array_split(x - x.mean(), 8)
        sds = np.array([np.std(s) for s in segs])
        ratio = float(sds.max() / (sds.min() + 1e-300))
        m["std_ratio"] = ratio
        if ratio > 5:
            add("nonstationary", "info",
                f"Amplitude varies {ratio:.0f}× between segments — non-stationary. "
                f"Prefer stft()/wavelet() over a single fft().", ratio)

    # ── 8. Signal-specific checks ─────────────────────────────────────────
    if signal_type == "ecg" and fs:
        _ecg_checks(x, fs, rep, add)

    return rep


# ECG signal-quality rules after Orphanidou et al., "Signal-quality indices for
# the electrocardiogram and photoplethysmogram", IEEE JBHI 19(3), 2015.
ECG_HR_RANGE = (40.0, 180.0)   # bpm
ECG_MAX_GAP = 3.0              # s between beats
ECG_RR_RATIO = 2.2             # max RR / min RR → 'irregular' note (info)
ECG_TEMPLATE_CORR = 0.66       # median beat-to-template correlation


def _ecg_checks(x: np.ndarray, fs: float, rep: QualityReport, add) -> None:
    """Heartbeat plausibility + beat-morphology consistency."""
    from scipy.signal import butter, sosfiltfilt

    from PBstats.features.ecg import pan_tompkins

    m = rep.metrics
    if fs < 50 or len(x) < 3 * fs:
        return
    # Judge morphology on the diagnostic band so drift / mains (reported by
    # their own checks) don't double-count as "noise" here.
    hi = min(40.0, 0.45 * fs)
    xf = sosfiltfilt(butter(2, [0.5, hi], btype="bandpass", fs=fs, output="sos"), x)
    r = pan_tompkins(xf, fs)
    dur = len(x) / fs
    m["ecg_beats"] = int(len(r))
    if len(r) < 3:
        add("ecg_beats", "critical",
            f"Only {len(r)} heartbeat(s) found in {dur:.0f} s — not a usable ECG "
            f"(lead off, heavy noise, or not an ECG).", float(len(r)))
        return

    rr = np.diff(r) / fs
    hr = 60.0 / rr.mean()
    m["ecg_hr_bpm"] = float(hr)
    m["ecg_rr_ratio"] = float(rr.max() / rr.min())
    problems = []
    if not ECG_HR_RANGE[0] <= hr <= ECG_HR_RANGE[1]:
        problems.append(f"mean HR {hr:.0f} bpm")
    if rr.max() > ECG_MAX_GAP:
        problems.append(f"{rr.max():.1f} s without a beat")
    if problems:
        add("ecg_rhythm", "warning",
            "Implausible beat sequence (" + "; ".join(problems) + ") — beats lost in noise, "
            "lead-off, or a genuine pause/extreme rate.", float(hr))
    elif m["ecg_rr_ratio"] > ECG_RR_RATIO:
        # Irregular rhythm is clinical content (ectopics, AF), not a quality defect.
        add("ecg_irregular", "info",
            f"Irregular RR intervals (max/min {m['ecg_rr_ratio']:.1f}) — ectopic beats or "
            f"arrhythmia, or occasional missed/extra detections.", m["ecg_rr_ratio"])

    half = int(np.median(rr) * fs / 2)
    beats = np.array([xf[i - half:i + half] for i in r if i - half >= 0 and i + half <= len(xf)])
    if len(beats) >= 3 and half > 2:
        # Median template + median correlation: robust to a minority of
        # differently shaped beats (PVCs), still sensitive to noise, which
        # corrupts most beats.
        template = np.median(beats, axis=0)
        tc = beats - beats.mean(axis=1, keepdims=True)
        tt = template - template.mean()
        corr = (tc @ tt) / (np.linalg.norm(tc, axis=1) * np.linalg.norm(tt) + 1e-300)
        mc = float(np.median(corr))
        m["ecg_template_corr"] = mc
        if mc < ECG_TEMPLATE_CORR:
            add("ecg_morphology", "warning",
                f"Beats are inconsistent in shape (median correlation with the median beat "
                f"{mc:.2f} < {ECG_TEMPLATE_CORR}) — in-band noise such as muscle or motion "
                f"artefact, which filtering cannot remove.", mc)


def _trend_check(x, rep, add) -> None:
    t = np.arange(len(x), dtype=float)
    r = np.corrcoef(t, x)[0, 1]
    r2 = float(r * r) if np.isfinite(r) else 0.0
    rep.metrics["trend_r2"] = r2
    if r2 > 0.3:
        add("trend", "warning",
            f"Strong linear trend (R² = {r2:.2f}) — detrend before spectral analysis.",
            r2, ("detrend", {"method": "linear"}))


def _line_db(f, p, f0) -> float:
    pk = (f >= f0 - 1) & (f <= f0 + 1)
    bg = ((f >= f0 - 10) & (f <= f0 - 3)) | ((f >= f0 + 3) & (f <= f0 + 10))
    if not pk.any() or bg.sum() < 3:
        return 0.0
    return float(10 * np.log10(p[pk].max() / (np.median(p[bg]) + 1e-300)))


# ═════════════════════════════════════════════════════════════════════════
# Mixin
# ═════════════════════════════════════════════════════════════════════════

class QualityMixin:

    def diagnose(
        self,
        mains: float | None = None,
        drift_cutoff: float | None = 0.5,
        verbose: bool = False,
        signal_type: str | None = None,
    ) -> "QualityMixin":
        """
        Inspect the signal and grade its quality. Result in
        ``self.quality_report`` (score, issues, metrics, prescription).
        Non-destructive.

        Parameters
        ----------
        mains        : power-line frequency to check (50 or 60). Default:
                       check both.
        drift_cutoff : frequency (Hz) below which power counts as baseline
                       drift (default 0.5, suited to ECG/EEG/EMG). None
                       disables the drift check.
        verbose      : print the report
        signal_type  : 'ecg' adds heartbeat checks — plausible heart rate /
                       RR sequence, and beat-shape consistency (detects
                       in-band muscle/motion noise that filters can't remove)

        Example
        -------
        d = Data(raw_ecg, fs=500).diagnose(signal_type="ecg", verbose=True)
        d.quality_report.pipeline.save("ecg_clean.json")
        """
        x = require_1d(self.data, "diagnose")
        self.quality_report = diagnose_signal(
            x, self.fs, self.label, mains=mains, drift_cutoff=drift_cutoff,
            signal_type=signal_type,
        )
        if verbose:
            self.quality_report.summary()
        return self

    def quality_timeline(
        self,
        window: float = 10.0,
        step: float | None = None,
        **diagnose_kwargs,
    ) -> pd.DataFrame:
        """
        Diagnose the recording in sliding windows — *where* is it bad?
        A long recording is rarely uniformly good or bad; this finds the
        stretches to discard or inspect.

        Parameters
        ----------
        window : window length in seconds (samples without fs)
        step   : hop (default = window)
        diagnose_kwargs : passed to ``diagnose`` (mains, drift_cutoff, signal_type)

        Returns
        -------
        DataFrame: t_start, t_end, score, grade, ok, issues (comma-separated codes).
        """
        x = require_1d(self.data, "quality_timeline")
        scale = self.fs or 1.0
        win = int(round(window * scale))
        hop = int(round((step or window) * scale))
        if win < 32 or win > len(x):
            raise ValueError(f"window must span 32..{len(x)} samples, got {win}")
        rows = []
        for i in range(0, len(x) - win + 1, hop):
            rep = diagnose_signal(x[i:i + win], self.fs, self.label, **diagnose_kwargs)
            rows.append({
                "t_start": i / scale, "t_end": (i + win) / scale,
                "score": rep.score, "grade": rep.grade, "ok": rep.ok,
                "issues": ",".join(sorted({iss.code for iss in rep.issues if iss.severity != "info"})),
            })
        self.quality_timeline_result = pd.DataFrame(rows)
        return self.quality_timeline_result

    @step
    def auto_clean(
        self,
        mains: float | None = None,
        drift_cutoff: float | None = 0.5,
        verbose: bool = False,
    ) -> "QualityMixin":
        """
        Diagnose the signal, then apply the prescribed fixes in a safe order
        (gaps → spikes → drift → line noise → offset). The before/after
        reports are kept in ``self.quality_report`` and
        ``self.quality_report_after``.

        For fully reproducible batch processing, freeze the prescription
        instead: ``Data(x, fs).diagnose().quality_report.pipeline``.
        """
        self.diagnose(mains=mains, drift_cutoff=drift_cutoff)
        before = self.quality_report
        for name, kwargs in before.fixes:
            getattr(self, name)(**kwargs)
        after = diagnose_signal(self.data, self.fs, self.label,
                                mains=mains, drift_cutoff=drift_cutoff)
        self.quality_report = before
        self.quality_report_after = after
        applied = ", ".join(n for n, _ in before.fixes) or "nothing to fix"
        self._record(f"auto_clean(score {before.score} → {after.score}; {applied})")
        if verbose:
            before.summary()
            safe_print(f"  After auto_clean: {after.score}/100 (grade {after.grade})")
        return self
