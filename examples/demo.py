"""
PBstats end-to-end demo: raw, messy ECG → diagnosis → cleaning →
reproducible pipeline → features → machine-learning model.

Run:  python examples/demo.py              (figures saved to ./demo_output)
      python examples/demo.py --show       (open figures interactively)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from PBstats import Data, Pipeline, SignalFeatureExtractor, extract_features

FS = 500


def synth_ecg(dur=30.0, hr=70.0, irregular=0.05, seed=0, noise=0.01):
    """Synthetic ECG (P-QRS-T Gaussians). `irregular` = RR-interval jitter;
    large values mimic atrial-fibrillation-like irregular rhythm."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS * dur)) / FS
    x = np.zeros_like(t)
    beat = 0.5
    waves = [(0.12, -0.2, 0.025), (-0.1, -0.03, 0.008), (1.0, 0.0, 0.01),
             (-0.15, 0.03, 0.01), (0.3, 0.25, 0.04)]
    while beat < dur:
        for a, mu, sig in waves:
            x += a * np.exp(-((t - beat - mu) ** 2) / (2 * sig ** 2))
        beat += 60.0 / hr * (1 + irregular * rng.standard_normal())
    return t, x + noise * rng.standard_normal(len(t))


def corrupt(t, x, seed=0):
    """What real recordings look like: mains hum, baseline wander,
    electrode-pop spikes, a dropout gap and a DC offset."""
    rng = np.random.default_rng(seed)
    y = x + 0.25 * np.sin(2 * np.pi * 50 * t)            # 50 Hz power line
    y += 0.7 * np.sin(2 * np.pi * 0.12 * t + rng.uniform(0, 6))  # breathing wander
    y += 1.5                                              # DC offset
    y[rng.choice(len(y), 6, replace=False)] += rng.choice([-4, 4], 6)  # spikes
    start = rng.integers(2000, len(y) - 200)
    y[start:start + 60] = np.nan                          # dropout
    return y


def main(out: Path, show: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    save = (lambda name: None) if show else (lambda name: str(out / name))

    # ── 1. Diagnose a messy recording ─────────────────────────────────────
    t, clean = synth_ecg(seed=1)
    raw = corrupt(t, clean, seed=1)
    d = Data(raw, fs=FS, label="patient_001").diagnose(verbose=True)
    d.plot_quality(save_path=save("1_diagnosis.png"))

    # ── 2. Apply the prescription ─────────────────────────────────────────
    d.auto_clean()
    print(f"Quality after auto_clean: {d.quality_report_after.score}/100  "
          f"(correlation with ground truth: {np.corrcoef(d.data, clean)[0, 1]:.3f})")
    d.copy().plot(save_path=save("2_cleaned.png"))

    # ── 3. Freeze it into a reproducible pipeline and batch-apply ────────
    rx = d.quality_report.pipeline + Pipeline("finish").add("standardize")
    rx.summary()
    rx.save(str(out / "ecg_clean.json"))
    batch = [Data(corrupt(*synth_ecg(seed=s), seed=s), fs=FS, label=f"rec_{s}")
             for s in range(2, 6)]
    cleaned = Pipeline.load(str(out / "ecg_clean.json")).run_batch(batch)
    scores = [Data(c.data, fs=FS).diagnose().quality_report.score for c in cleaned]
    print(f"Batch of {len(cleaned)} recordings cleaned with the saved pipeline; "
          f"quality scores: {scores}")

    # ── 4. Analyse and extract features ───────────────────────────────────
    d.psd().describe()
    d.plot_psd(max_freq=60, save_path=save("3_psd.png"))
    d.auto_features(include=["time", "freq", "nonlinear", "hrv"])
    f = d.feature_vector.features
    print(f"\nHR {f['hrv_mean_hr_bpm']:.1f} bpm | SDNN {f['hrv_sdnn_ms']:.1f} ms | "
          f"RMSSD {f['hrv_rmssd_ms']:.1f} ms | SampEn {f['nl_sample_entropy']:.3f} | "
          f"DFA α {f['nl_dfa_alpha']:.2f}   ({len(f)} features total)")
    d.dashboard(save_path=save("4_dashboard.png"))

    # ── 5. Machine learning: regular vs. irregular rhythm ─────────────────
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    X, y = [], []
    for s in range(40):
        irregular = s % 2 == 1
        _, sig = synth_ecg(dur=30, hr=np.random.default_rng(s).uniform(60, 90),
                           irregular=0.18 if irregular else 0.03, seed=100 + s)
        X.append(sig)
        y.append(int(irregular))
    X, y = np.array(X), np.array(y)

    model = make_pipeline(
        SignalFeatureExtractor(fs=FS, include=["hrv"],
                               preprocess=Pipeline("clean").add("bandpass", low=0.5, high=40)),
        SimpleImputer(),
        RandomForestClassifier(n_estimators=200, random_state=0),
    )
    acc = cross_val_score(model, X, y, cv=5)
    print(f"\nRegular vs irregular rhythm, 5-fold CV accuracy: {acc.mean():.2f} ± {acc.std():.2f}")

    table = extract_features(X[:6], fs=FS, labels=y[:6], include=["hrv"])
    print(table[["label", "hrv_mean_hr_bpm", "hrv_rmssd_ms", "hrv_sd1_sd2"]].round(2))

    if not show:
        print(f"\nFigures and pipeline written to {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="demo_output", type=Path)
    ap.add_argument("--show", action="store_true", help="show figures instead of saving")
    args = ap.parse_args()
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    main(args.out, args.show)
