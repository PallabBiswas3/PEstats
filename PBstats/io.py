"""
Readers for common signal file formats — pure Python/NumPy, no extra
dependencies.

    Data.from_csv("log.csv", time_col="timestamp")      # fs inferred
    Data.from_wav("voice.wav")
    Data.from_edf("sleep.edf", channels=["EEG Fpz-Cz"])  # EDF, EDF+, BDF
    Data.from_wfdb("mitdb/100")                          # PhysioNet WFDB (local)
    Data.from_physionet("mitdb/100")                     # download + cache + read

Annotations found in the file (WFDB beat labels, EDF+ events) are attached as
``d.annotations``: a DataFrame with ``onset`` (s), ``duration`` (s), ``label``
and ``aux`` columns — ready for ``d.epochs(...)``.
"""
from __future__ import annotations

import os
import re
import urllib.request
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from PBstats.core.data import Data

ANNOTATION_COLUMNS = ["onset", "duration", "label", "aux"]


def _empty_annotations() -> pd.DataFrame:
    return pd.DataFrame(columns=ANNOTATION_COLUMNS).astype(
        {"onset": float, "duration": float, "label": object, "aux": object}
    )


def _select(names: list[str], channels) -> list[int]:
    if channels is None:
        return list(range(len(names)))
    idx = []
    for c in ([channels] if isinstance(channels, (str, int)) else channels):
        if isinstance(c, str):
            if c not in names:
                raise KeyError(f"Channel {c!r} not found. Available: {names}")
            idx.append(names.index(c))
        else:
            idx.append(int(c))
    return idx


def _make(data, fs, label, names, annotations=None, meta=None) -> "Data":
    from PBstats.core.data import Data
    arr = np.asarray(data, dtype=float)
    d = Data(arr, fs=fs, label=label,
             channel_names=names if arr.ndim == 2 and arr.shape[1] > 1 else None,
             annotations=annotations if annotations is not None and len(annotations) else None)
    d.meta = meta or {}
    return d


# ═════════════════════════════════════════════════════════════════════════
# CSV
# ═════════════════════════════════════════════════════════════════════════

def read_csv(
    path,
    time_col: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
    fs: Optional[float] = None,
    label: Optional[str] = None,
    regularize: bool = False,
    **read_kwargs,
) -> "Data":
    """
    Read a CSV of samples (rows) × channels (columns).

    time_col   : column with timestamps (seconds or anything pandas parses as
                 datetimes). The sampling rate is inferred from the median
                 interval. Irregular sampling triggers a warning, or — with
                 ``regularize=True`` — linear interpolation onto a uniform grid.
    columns    : channels to keep (default: every numeric column)
    fs         : sampling rate, if there is no time column
    read_kwargs: passed to ``pandas.read_csv`` (sep, skiprows, ...)
    """
    df = pd.read_csv(path, **read_kwargs)
    meta = {"source": str(path)}

    t = None
    if time_col is not None:
        raw_t = df.pop(time_col)
        if pd.api.types.is_numeric_dtype(raw_t):
            t = raw_t.to_numpy(dtype=float)
        else:
            dt = pd.to_datetime(raw_t)
            meta["start_time"] = dt.iloc[0]
            t = (dt - dt.iloc[0]).dt.total_seconds().to_numpy()
        steps = np.diff(t)
        if len(steps) == 0 or np.any(steps <= 0):
            raise ValueError(f"Time column {time_col!r} must be strictly increasing")
        step = float(np.median(steps))
        inferred = 1.0 / step
        if fs is not None and abs(fs - inferred) / inferred > 0.01:
            warnings.warn(f"fs={fs} disagrees with timestamps (≈{inferred:.4g} Hz); using fs={fs}",
                          stacklevel=2)
        else:
            fs = inferred
        jitter = float(np.max(np.abs(steps - step)) / step)
        meta["timing_jitter"] = jitter
        if jitter > 0.01 and not regularize:
            warnings.warn(
                f"Irregular sampling: intervals deviate up to {jitter:.0%} from the median. "
                f"Pass regularize=True to interpolate onto a uniform {fs:.4g} Hz grid.",
                stacklevel=2,
            )

    if columns is not None:
        df = df[list(columns)]
    df = df.select_dtypes(include="number")
    if df.shape[1] == 0:
        raise ValueError("No numeric columns found")
    values = df.to_numpy(dtype=float)

    if regularize and t is not None:
        grid = np.arange(t[0], t[-1] + 0.5 / fs, 1.0 / fs)
        values = np.column_stack([np.interp(grid, t, values[:, c]) for c in range(values.shape[1])])

    return _make(values, fs, label if label is not None else Path(path).stem,
                 [str(c) for c in df.columns], meta=meta)


# ═════════════════════════════════════════════════════════════════════════
# WAV
# ═════════════════════════════════════════════════════════════════════════

def read_wav(path, label: Optional[str] = None) -> "Data":
    """Read a WAV file; integer PCM is scaled to [-1, 1)."""
    from scipy.io import wavfile

    fs, x = wavfile.read(path)
    if x.dtype == np.uint8:
        x = (x.astype(float) - 128) / 128
    elif np.issubdtype(x.dtype, np.integer):
        x = x.astype(float) / float(np.iinfo(x.dtype).max + 1)
    else:
        x = x.astype(float)
    names = ["left", "right"] if x.ndim == 2 and x.shape[1] == 2 else (
        [f"ch{i}" for i in range(x.shape[1])] if x.ndim == 2 else None)
    return _make(x, float(fs), label if label is not None else Path(path).stem, names,
                 meta={"source": str(path)})


# ═════════════════════════════════════════════════════════════════════════
# EDF / EDF+ / BDF
# ═════════════════════════════════════════════════════════════════════════

def _parse_tal(raw: bytes) -> list[tuple[float, float, str]]:
    """Parse EDF+ Time-stamped Annotation Lists."""
    out = []
    for tal in raw.split(b"\x00"):
        if not tal.strip(b"\x00"):
            continue
        parts = tal.split(b"\x14")
        head = parts[0].split(b"\x15")
        try:
            onset = float(head[0])
        except ValueError:
            continue
        duration = float(head[1]) if len(head) > 1 and head[1] else 0.0
        for text in parts[1:]:
            if text:
                out.append((onset, duration, text.decode("latin-1")))
    return out


def read_edf(path, channels=None, label: Optional[str] = None) -> "Data":
    """
    Read EDF, EDF+ (incl. annotations) or BDF (24-bit BioSemi) files.

    channels : names or indices to load. Channels in one Data object must
               share a sampling rate; by default the channels with the most
               common rate are loaded (with a warning if others are skipped).
    """
    with open(path, "rb") as f:
        raw = f.read()

    def field(start, length):
        return raw[start:start + length].decode("latin-1").strip()

    bdf = raw[:1] == b"\xff"
    nbytes = 3 if bdf else 2
    header_bytes = int(field(184, 8))
    n_records = int(field(236, 8))
    rec_dur = float(field(244, 8))
    ns = int(field(252, 4))

    def sig_fields(offset, width, cast=str):
        base = 256 + offset * ns
        return [cast(raw[base + i * width: base + (i + 1) * width].decode("latin-1").strip())
                for i in range(ns)]

    labels = sig_fields(0, 16)
    units = sig_fields(16 + 80, 8)
    off = 16 + 80 + 8
    pmin = sig_fields(off, 8, float)
    pmax = sig_fields(off + 8, 8, float)
    dmin = sig_fields(off + 16, 8, float)
    dmax = sig_fields(off + 24, 8, float)
    spr = sig_fields(off + 32 + 80, 8, int)

    rec_size = sum(spr) * nbytes
    if n_records < 0:
        n_records = (len(raw) - header_bytes) // rec_size
    body = np.frombuffer(raw, dtype=np.uint8, count=n_records * rec_size, offset=header_bytes)
    body = body.reshape(n_records, rec_size)

    annot_idx = [i for i, lab in enumerate(labels) if lab in ("EDF Annotations", "BDF Annotations")]
    data_idx = [i for i in range(ns) if i not in annot_idx]
    data_names = [labels[i] for i in data_idx]

    # Annotations
    starts = np.concatenate([[0], np.cumsum(spr)]) * nbytes
    ann = []
    for i in annot_idx:
        ann += _parse_tal(body[:, starts[i]:starts[i + 1]].tobytes())
    annotations = _empty_annotations()
    if ann:
        annotations = pd.DataFrame(
            [(o, d, t, "") for o, d, t in ann if t.strip()], columns=ANNOTATION_COLUMNS,
        ).sort_values("onset", ignore_index=True)

    # Channel selection
    rates = {i: spr[i] / rec_dur for i in data_idx}
    if channels is None:
        values, counts = np.unique([rates[i] for i in data_idx], return_counts=True)
        common = values[np.argmax(counts)]
        chosen = [i for i in data_idx if rates[i] == common]
        skipped = [labels[i] for i in data_idx if rates[i] != common]
        if skipped:
            warnings.warn(f"Skipped channels with a different sampling rate: {skipped}. "
                          f"Load them separately with channels=[...].", stacklevel=2)
    else:
        chosen = [data_idx[j] for j in _select(data_names, channels)]
    if len({rates[i] for i in chosen}) > 1:
        raise ValueError("Selected channels have different sampling rates; load them separately.")

    cols = []
    for i in chosen:
        block = body[:, starts[i]:starts[i + 1]].reshape(-1)
        if bdf:
            b = block.reshape(-1, 3).astype(np.int32)
            dig = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            dig = np.where(dig >= 1 << 23, dig - (1 << 24), dig)
        else:
            dig = block.view("<i2")
        scale = (pmax[i] - pmin[i]) / (dmax[i] - dmin[i])
        cols.append((dig.astype(float) - dmin[i]) * scale + pmin[i])

    meta = {"source": str(path), "units": [units[i] for i in chosen],
            "patient": field(8, 80), "recording": field(88, 80),
            "start": f"{field(168, 8)} {field(176, 8)}"}
    return _make(np.column_stack(cols), rates[chosen[0]],
                 label if label is not None else Path(path).stem,
                 [labels[i] for i in chosen], annotations, meta)


# ═════════════════════════════════════════════════════════════════════════
# WFDB (PhysioNet)
# ═════════════════════════════════════════════════════════════════════════

_ANN_SYMBOLS = {
    1: "N", 2: "L", 3: "R", 4: "a", 5: "V", 6: "F", 7: "J", 8: "A", 9: "S", 10: "E",
    11: "j", 12: "/", 13: "Q", 14: "~", 16: "|", 18: "s", 19: "T", 20: "*", 21: "D",
    22: '"', 23: "=", 24: "p", 25: "B", 26: "^", 27: "t", 28: "+", 29: "u", 30: "?",
    31: "!", 32: "[", 33: "]", 34: "e", 35: "n", 36: "@", 37: "x", 38: "f", 39: "(",
    40: ")", 41: "r",
}
#: Annotation symbols that mark a heartbeat (PhysioNet convention).
BEAT_SYMBOLS = frozenset("NLRBAaJSVrFejnE/fQ?")


def _parse_header(hea: Path) -> dict:
    lines = [ln.strip() for ln in hea.read_text(encoding="latin-1").splitlines()]
    comments = [ln[1:].strip() for ln in lines if ln.startswith("#")]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    rec = lines[0].split()
    if "/" in rec[0]:
        raise NotImplementedError("Multi-segment WFDB records are not supported")
    nsig = int(rec[1])
    fs = float(re.split(r"[/(]", rec[2])[0]) if len(rec) > 2 else 250.0
    nsamp = int(rec[3]) if len(rec) > 3 else None
    sigs = []
    for ln in lines[1:1 + nsig]:
        p = ln.split()
        m = re.match(r"(\d+)(?:x\d+)?(?::\d+)?(?:\+(\d+))?", p[1])
        gm = re.match(r"([-\d.eE+]+)?(?:\(([-\d]+)\))?(?:/(\S+))?", p[2]) if len(p) > 2 else None
        gain = float(gm.group(1)) if gm and gm.group(1) else 200.0
        adczero = int(p[4]) if len(p) > 4 else 0
        baseline = int(gm.group(2)) if gm and gm.group(2) else adczero
        sigs.append({
            "file": p[0], "fmt": int(m.group(1)), "offset": int(m.group(2) or 0),
            "gain": gain or 200.0, "baseline": baseline,
            "units": (gm.group(3) if gm and gm.group(3) else "mV"),
            "name": " ".join(p[8:]) if len(p) > 8 else f"sig{len(sigs)}",
        })
    return {"nsig": nsig, "fs": fs, "nsamp": nsamp, "signals": sigs, "comments": comments}


def _decode(raw: bytes, fmt: int, n_frames: Optional[int], n_sig: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (digital samples (frames, n_sig), invalid mask)."""
    if fmt == 212:
        b = np.frombuffer(raw[: len(raw) // 3 * 3], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        s = np.empty(len(b) * 2, dtype=np.int32)
        s[0::2] = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
        s[1::2] = b[:, 2] | ((b[:, 1] & 0xF0) << 4)
        s = np.where(s > 2047, s - 4096, s)
        invalid = s == -2048
    elif fmt == 16:
        s = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.int32)
        invalid = s == -32768
    elif fmt == 61:
        s = np.frombuffer(raw[: len(raw) // 2 * 2], dtype=">i2").astype(np.int32)
        invalid = s == -32768
    elif fmt == 80:
        s = np.frombuffer(raw, dtype=np.uint8).astype(np.int32) - 128
        invalid = s == -128
    elif fmt == 24:
        b = np.frombuffer(raw[: len(raw) // 3 * 3], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        s = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        s = np.where(s >= 1 << 23, s - (1 << 24), s)
        invalid = s == -(1 << 23)
    elif fmt == 32:
        s = np.frombuffer(raw[: len(raw) // 4 * 4], dtype="<i4").astype(np.int64)
        invalid = s == -(1 << 31)
    else:
        raise NotImplementedError(f"WFDB format {fmt} is not supported (supported: 16, 24, 32, 61, 80, 212)")
    n = (len(s) // n_sig) if n_frames is None else min(n_frames, len(s) // n_sig)
    return s[: n * n_sig].reshape(n, n_sig), invalid[: n * n_sig].reshape(n, n_sig)


def read_annotations(record, extension: str = "atr", fs: Optional[float] = None) -> pd.DataFrame:
    """
    Read a WFDB (MIT-format) annotation file → DataFrame with ``sample``,
    ``onset`` (s), ``duration``, ``label`` (symbol) and ``aux`` (e.g. rhythm
    "(AFIB"). Beat annotations: ``df[df.label.isin(BEAT_SYMBOLS)]``.
    """
    record = Path(record)
    if fs is None:
        fs = _parse_header(record.with_suffix(".hea"))["fs"]
    w = np.frombuffer(record.with_suffix("." + extension).read_bytes(), dtype="<u2")
    rows, sample, i = [], 0, 0
    while i < len(w):
        a, v = int(w[i] >> 10), int(w[i] & 0x3FF)
        i += 1
        if a == 0 and v == 0:
            break
        if a == 59:                                   # SKIP: 32-bit interval follows
            if i + 1 < len(w):
                interval = (int(w[i]) << 16) | int(w[i + 1])
                if interval >= 1 << 31:
                    interval -= 1 << 32
                sample += interval
            i += 2
        elif a in (60, 61, 62):                       # NUM, SUB, CHN
            continue
        elif a == 63:                                 # AUX string for the previous annotation
            text = w[i:i + (v + 1) // 2].tobytes()[:v].decode("latin-1").rstrip("\x00")
            if rows:
                rows[-1][4] = text
            i += (v + 1) // 2
        else:
            sample += v
            rows.append([sample, sample / fs, 0.0, _ANN_SYMBOLS.get(a, str(a)), ""])
    return pd.DataFrame(rows, columns=["sample", "onset", "duration", "label", "aux"])


def read_wfdb(
    record,
    channels=None,
    start: float = 0.0,
    end: Optional[float] = None,
    annotations: Optional[str] = "atr",
    label: Optional[str] = None,
) -> "Data":
    """
    Read a local PhysioNet WFDB record (``.hea`` + ``.dat``), physical units.

    record      : path without extension, e.g. "data/mitdb/100"
    channels    : names (e.g. "MLII") or indices; default all
    start / end : time window in seconds
    annotations : annotation extension to attach if present ("atr"), or None
    """
    record = Path(record)
    hdr = _parse_header(record.with_suffix(".hea"))
    fs, sigs = hdr["fs"], hdr["signals"]
    names = [s["name"] for s in sigs]
    chosen = _select(names, channels)

    # Group signals by .dat file (interleaved within a file).
    columns: dict[int, np.ndarray] = {}
    files: dict[str, list[int]] = {}
    for k, s in enumerate(sigs):
        files.setdefault(s["file"], []).append(k)
    for fname, members in files.items():
        if not any(k in chosen for k in members):
            continue
        fmt, offset = sigs[members[0]]["fmt"], sigs[members[0]]["offset"]
        raw = (record.parent / fname).read_bytes()[offset:]
        dig, bad = _decode(raw, fmt, hdr["nsamp"], len(members))
        for j, k in enumerate(members):
            col = (dig[:, j] - sigs[k]["baseline"]) / sigs[k]["gain"]
            col[bad[:, j]] = np.nan
            columns[k] = col

    x = np.column_stack([columns[k] for k in chosen])
    i0 = int(round(start * fs))
    i1 = len(x) if end is None else int(round(end * fs))
    x = x[i0:i1]

    ann = _empty_annotations()
    ann_path = record.with_suffix("." + annotations) if annotations else None
    if ann_path is not None and ann_path.exists():
        a = read_annotations(record, annotations, fs)
        a = a[(a["sample"] >= i0) & (a["sample"] < i0 + len(x))].copy()
        a["onset"] = (a["sample"] - i0) / fs
        ann = a[ANNOTATION_COLUMNS].reset_index(drop=True)

    meta = {"source": str(record), "units": [sigs[k]["units"] for k in chosen],
            "comments": hdr["comments"], "start_s": start}
    return _make(x, fs, label if label is not None else record.name,
                 [names[k] for k in chosen], ann, meta)


def default_cache_dir() -> Path:
    return Path(os.environ.get("PBSTATS_DATA", Path.home() / ".cache" / "pbstats"))


def download_physionet(
    record: str, version: str = "1.0.0", cache_dir=None,
    extensions: Sequence[str] = ("hea", "dat", "atr"),
) -> Path:
    """
    Download a PhysioNet record (e.g. "mitdb/100") into a local cache
    (``~/.cache/pbstats`` or ``$PBSTATS_DATA``) and return its path stem.
    Files already cached are not downloaded again. The data's own licence
    (usually ODC-By) applies.
    """
    db, name = record.strip("/").split("/", 1)
    dest = Path(cache_dir or default_cache_dir()) / db
    dest.mkdir(parents=True, exist_ok=True)
    base = f"https://physionet.org/files/{db}/{version}/"
    header = dest / f"{name}.hea"
    if not header.exists():
        urllib.request.urlretrieve(base + f"{name}.hea", header)
    dat_files = {s["file"] for s in _parse_header(header)["signals"]}
    wanted = [f"{name}.{e}" for e in extensions if e not in ("hea", "dat")] + sorted(dat_files)
    for fname in wanted:
        target = dest / fname
        if target.exists() and target.stat().st_size > 0:
            continue
        try:
            urllib.request.urlretrieve(base + fname, target)
        except urllib.error.HTTPError as e:
            if fname.endswith(".atr") and e.code == 404:
                continue          # not every record has annotations
            raise
    return dest / name


def read_physionet(record: str, version: str = "1.0.0", cache_dir=None, **kwargs) -> "Data":
    """Download (once, cached) and read a PhysioNet record, e.g. "mitdb/100"."""
    return read_wfdb(download_physionet(record, version, cache_dir), **kwargs)
