"""
R-peak detection benchmark on the MIT-BIH Arrhythmia Database (all 48
records, channel 0, reference = beat annotations, ±150 ms tolerance).

    python benchmarks/rpeaks_mitdb.py                 # downloads ~100 MB once
    python benchmarks/rpeaks_mitdb.py --data-dir D    # use files already in D/mitdb

Data: Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
IEEE Eng in Med and Biol 20(3):45-50 (2001). PhysioNet, ODC-By licence.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from PBstats.features.ecg import detect_r_peaks, match_beats
from PBstats.io import BEAT_SYMBOLS, download_physionet, read_wfdb

RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "111", "112",
    "113", "114", "115", "116", "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210", "212", "213", "214", "215",
    "217", "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234",
]


def run(data_dir: Path | None) -> pd.DataFrame:
    rows = []
    for rec in RECORDS:
        path = (data_dir / "mitdb" / rec) if data_dir else download_physionet(f"mitdb/{rec}")
        d = read_wfdb(path, channels=0)
        ref = (d.annotations[d.annotations["label"].isin(BEAT_SYMBOLS)]["onset"] * d.fs).round().astype(int)
        for method in ("pantompkins", "simple"):
            t0 = time.perf_counter()
            det = detect_r_peaks(d.data, d.fs, method=method)
            scores = match_beats(ref.to_numpy(), det, d.fs)
            rows.append({"record": rec, "method": method, **scores,
                         "seconds": time.perf_counter() - t0})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("method")[["TP", "FN", "FP"]].sum()
    g["Se_%"] = 100 * g.TP / (g.TP + g.FN)
    g["PPV_%"] = 100 * g.TP / (g.TP + g.FP)
    g["F1_%"] = 2 * g["Se_%"] * g["PPV_%"] / (g["Se_%"] + g["PPV_%"])
    return g.round(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = ap.parse_args()

    df = run(args.data_dir)
    summary = summarise(df)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "rpeaks_mitdb_per_record.csv", index=False)
    (args.out / "rpeaks_mitdb_summary.json").write_text(json.dumps(summary.to_dict("index"), indent=2))
    print(summary.to_string())
    worst = df[df.method == "pantompkins"].nsmallest(5, "F1")[["record", "Se", "PPV", "F1"]]
    print("\nHardest records for Pan-Tompkins:\n", worst.round(4).to_string(index=False))
