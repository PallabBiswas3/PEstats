"""Minimal spec-based writers used to test the readers without extra
dependencies. (The readers were also cross-checked bit-for-bit against the
reference `wfdb` and `pyedflib` libraries during development.)"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_SYMBOL_CODES = {"N": 1, "V": 5, "A": 8, "+": 28, "~": 14}


def write_edf(path, signals: dict, fs: float, annotations=(), record_seconds: float = 1.0):
    """EDF+ writer: signals {label: array (physical units)}; annotations
    [(onset_s, duration_s, text)] go in an 'EDF Annotations' channel."""
    labels = list(signals)
    n = len(next(iter(signals.values())))
    spr = int(fs * record_seconds)
    n_rec = n // spr
    ann_spr = 64  # int16 slots per record for TALs
    ns = len(labels) + 1

    def pad(s, w):
        return str(s)[:w].ljust(w).encode("latin-1")

    pmin = [float(np.min(signals[k])) - 1 for k in labels]
    pmax = [float(np.max(signals[k])) + 1 for k in labels]
    hdr = (pad("0", 8) + pad("X X X X", 80) + pad("Startdate X X X X", 80)
           + pad("01.01.26", 8) + pad("00.00.00", 8) + pad(256 * (ns + 1), 8)
           + pad("EDF+C", 44) + pad(n_rec, 8) + pad(record_seconds, 8) + pad(ns, 4))
    fields = [
        [pad(k, 16) for k in labels] + [pad("EDF Annotations", 16)],
        [pad("", 80)] * ns,
        [pad("uV", 8)] * len(labels) + [pad("", 8)],
        [pad(f"{v:.3f}", 8) for v in pmin] + [pad("-1", 8)],
        [pad(f"{v:.3f}", 8) for v in pmax] + [pad("1", 8)],
        [pad("-32768", 8)] * ns,
        [pad("32767", 8)] * ns,
        [pad("", 80)] * ns,
        [pad(spr, 8)] * len(labels) + [pad(ann_spr, 8)],
        [pad("", 32)] * ns,
    ]
    for f in fields:
        hdr += b"".join(f)

    body = bytearray()
    for r in range(n_rec):
        for k, lo, hi in zip(labels, pmin, pmax):
            seg = np.asarray(signals[k][r * spr:(r + 1) * spr], dtype=float)
            dig = np.round((seg - lo) / (hi - lo) * 65535 - 32768).astype("<i2")
            body += dig.tobytes()
        tal = f"+{r * record_seconds:g}\x14\x14\x00".encode()
        for onset, dur, text in annotations:
            if r * record_seconds <= onset < (r + 1) * record_seconds:
                tal += f"+{onset:g}\x15{dur:g}\x14{text}\x14\x00".encode()
        body += tal.ljust(ann_spr * 2, b"\x00")
    Path(path).write_bytes(hdr + bytes(body))
    return pmin, pmax


def write_wfdb(stem, signal: np.ndarray, fs: float, fmt: int = 212, gain: float = 200.0,
               names=("MLII", "V5"), annotations=()):
    """Write a 2-channel WFDB record (.hea/.dat) and optional .atr
    annotations [(sample, symbol)]. Returns the digital values written."""
    stem = Path(stem)
    dig = np.round(signal * gain).astype(np.int32)
    flat = dig.reshape(-1)
    if fmt == 16:
        raw = flat.astype("<i2").tobytes()
    elif fmt == 212:
        if len(flat) % 2:
            flat = np.r_[flat, 0]
        a, b = flat[0::2] & 0xFFF, flat[1::2] & 0xFFF
        out = np.empty((len(a), 3), dtype=np.uint8)
        out[:, 0] = a & 0xFF
        out[:, 1] = ((a >> 8) & 0x0F) | (((b >> 8) & 0x0F) << 4)
        out[:, 2] = b & 0xFF
        raw = out.tobytes()
    else:
        raise ValueError(fmt)
    stem.with_suffix(".dat").write_bytes(raw)
    lines = [f"{stem.name} {dig.shape[1]} {fs:g} {dig.shape[0]}"]
    for j, nm in enumerate(names[: dig.shape[1]]):
        lines.append(f"{stem.name}.dat {fmt} {gain:g}(0)/mV 12 0 {dig[0, j]} 0 0 {nm}")
    stem.with_suffix(".hea").write_text("\n".join(lines) + "\n")

    if annotations:
        words, prev = [], 0
        for sample, sym in annotations:
            delta = sample - prev
            if delta > 1023:
                words += [59 << 10, (delta >> 16) & 0xFFFF, delta & 0xFFFF]
                words.append(_SYMBOL_CODES[sym] << 10)
            else:
                words.append((_SYMBOL_CODES[sym] << 10) | delta)
            prev = sample
        words.append(0)
        stem.with_suffix(".atr").write_bytes(np.asarray(words, dtype="<u2").tobytes())
    return dig
