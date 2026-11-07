"""
Signal-quality benchmark on the MIT-BIH Noise Stress Test Database (NSTDB).

NSTDB records 118e* / 119e* are MIT-BIH records 118 and 119 with real
electrode-motion ("em") noise added at a known SNR (24 … -6 dB). Noise is
present in alternating 2-minute segments starting at minute 5, so every
10-second window has a ground-truth label (clean / noisy).

For each SNR we report, on channel 0:
  * detection rate  — share of noisy windows that diagnose() flags (not ok)
  * false alarms    — share of clean windows flagged (from the clean MIT-BIH
                      originals 118 and 119 — no added noise at all)
  * AUC             — how well the 0–100 score separates noisy from clean
  * R-peak F1       — beat detection under noise, Pan-Tompkins vs simple

    python benchmarks/quality_nstdb.py                 # downloads ~45 MB once
    python benchmarks/quality_nstdb.py --data-dir D    # D/mitdb, D/nstdb

Data: Moody GB, Muldrow WE, Mark RG. A noise stress test for arrhythmia
detectors. Computers in Cardiology 11:381-384 (1984). PhysioNet, ODC-By.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from PBstats.diagnostics.quality import diagnose_signal
from PBstats.features.ecg import detect_r_peaks, match_beats
from PBstats.io import BEAT_SYMBOLS, download_physionet, read_wfdb

SNRS = ["24", "18", "12", "06", "00", "_6"]
WINDOW = 10.0


def noisy_mask_label(t0: float, t1: float) -> str | None:
    """'noisy' / 'clean' if [t0, t1) lies fully inside one NSTDB segment."""
    def state(t):
        if t < 300:
            return "clean"
        return "noisy" if int((t - 300) // 120) % 2 == 0 else "clean"
    a, b = state(t0), state(t1 - 1e-6)
    return a if a == b else None


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) — Mann-Whitney AUC with ties counted half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size))


def windows(d, label_fn, signal_type):
    win = int(WINDOW * d.fs)
    rows = []
    for i in range(0, len(d.data) - win + 1, win):
        t0, t1 = i / d.fs, (i + win) / d.fs
        lab = label_fn(t0, t1)
        if lab is None:
            continue
        rep = diagnose_signal(d.data[i:i + win], d.fs, signal_type=signal_type)
        rows.append({"truth": lab, "badness": 100 - rep.score, "flagged": not rep.ok,
                     "issues": ",".join(sorted({x.code for x in rep.issues if x.severity != "info"}))})
    return pd.DataFrame(rows)


def beats_f1(d) -> dict:
    ref = (d.annotations[d.annotations["label"].isin(BEAT_SYMBOLS)]["onset"] * d.fs).round().astype(int)
    return {m: match_beats(ref.to_numpy(), detect_r_peaks(d.data, d.fs, method=m), d.fs)["F1"]
            for m in ("pantompkins", "simple")}


def load(db: str, rec: str, data_dir: Path | None):
    path = (data_dir / db / rec) if data_dir else download_physionet(f"{db}/{rec}")
    return read_wfdb(path, channels=0)


def run(data_dir: Path | None) -> pd.DataFrame:
    rows = []
    for signal_type in (None, "ecg"):
        clean_windows = pd.concat([
            windows(load("mitdb", base, data_dir), lambda a, b: "clean", signal_type)
            for base in ("118", "119")
        ])
        false_alarm = clean_windows["flagged"].mean()
        for snr in SNRS:
            parts, f1s = [], []
            for base in ("118", "119"):
                d = load("nstdb", f"{base}e{snr}", data_dir)
                w = windows(d, noisy_mask_label, signal_type)
                parts.append(w[w.truth == "noisy"])
                if signal_type is None:
                    f1s.append(beats_f1(d))
            noisy = pd.concat(parts)
            row = {
                "checks": "ecg" if signal_type else "generic",
                "snr_db": snr.replace("_", "-"),
                "detection_%": 100 * noisy["flagged"].mean(),
                "false_alarm_%": 100 * false_alarm,
                "auc": auc(noisy["badness"].to_numpy(), clean_windows["badness"].to_numpy()),
                "n_noisy": len(noisy), "n_clean": len(clean_windows),
            }
            if f1s:
                row["F1_pantompkins_%"] = 100 * np.mean([f["pantompkins"] for f in f1s])
                row["F1_simple_%"] = 100 * np.mean([f["simple"] for f in f1s])
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = ap.parse_args()
    df = run(args.data_dir).round(3)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "quality_nstdb.csv", index=False)
    (args.out / "quality_nstdb.json").write_text(json.dumps(df.to_dict("records"), indent=2))
    print(df.to_string(index=False))
