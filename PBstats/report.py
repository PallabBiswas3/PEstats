"""
One-file HTML report: ``d.report("patient_001.html")``.

A self-contained page (figures embedded, no external files or network) with:
quality score and issues, the prescribed cleaning pipeline, signal and
spectrum figures, a quality-over-time strip for long recordings, descriptive
statistics, extracted features, and the full processing history as a
replayable pipeline. Easy to email, attach to a ticket, or archive next to
the data.
"""
from __future__ import annotations

import base64
import datetime as _dt
import html
import io
import json
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from PBstats.core.data import Data

_CSS = """
:root{--bg:#fafaf9;--card:#fff;--ink:#1c1c1e;--muted:#6b6b70;--line:#e6e4ea;--accent:#5B4FCF;
--ok:#1D9E75;--warn:#BA7517;--bad:#C0392B;--code:#f3f2f7}
@media (prefers-color-scheme:dark){:root{--bg:#141417;--card:#1d1d22;--ink:#ececf1;--muted:#a0a0aa;
--line:#2f2f37;--code:#26262d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}
header h1{font-size:26px;margin:0 0 4px}header p{margin:0;color:var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 8px}
.chip{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:4px 12px;font-size:13px}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin-top:20px}
h2{font-size:17px;margin:0 0 14px}
.score{display:flex;align-items:center;gap:22px;flex-wrap:wrap}
.ring{width:96px;height:96px;border-radius:50%;display:grid;place-items:center;font-weight:700;font-size:26px}
.ring small{display:block;font-size:12px;font-weight:500;color:var(--muted);text-align:center}
ul.issues{list-style:none;margin:0;padding:0;flex:1;min-width:260px}
ul.issues li{padding:8px 0;border-bottom:1px solid var(--line)}ul.issues li:last-child{border:0}
.pill{display:inline-block;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;
border-radius:4px;padding:1px 6px;margin-right:8px;color:#fff}
.critical{background:var(--bad)}.warning{background:var(--warn)}.info{background:var(--muted)}
img{max-width:100%;height:auto;border-radius:8px;background:#fff;display:block;margin:6px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px}
td,th{text-align:left;padding:5px 10px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:4px 28px}
pre{background:var(--code);border-radius:8px;padding:12px 14px;overflow-x:auto;font-size:12.5px;margin:8px 0}
.muted{color:var(--muted)}footer{margin-top:28px;color:var(--muted);font-size:12.5px;text-align:center}
"""


def _fig_to_img(draw) -> str:
    """Run a plotting call that accepts save_path and embed the PNG.
    (Saving to a path never calls plt.show(), so no backend switch is needed —
    switching would close the user's other open figures.)"""
    buf = io.BytesIO()
    draw(buf)
    return f'<img alt="figure" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "—"
        return f"{v:.4g}"
    return html.escape(str(v))


def _table(rows: list[tuple], header=("Metric", "Value")) -> str:
    body = "".join(f"<tr><td>{html.escape(str(k))}</td><td class='num'>{_fmt(v)}</td></tr>" for k, v in rows)
    return f"<table><tr><th>{header[0]}</th><th style='text-align:right'>{header[1]}</th></tr>{body}</table>"


def build_report(
    d: "Data",
    path: Optional[str] = None,
    title: Optional[str] = None,
    signal_type: Optional[str] = None,
    channel=0,
    features: bool = True,
    max_feature_seconds: float = 300.0,
) -> str:
    """Render the report; write it to ``path`` if given. Returns the HTML."""
    from PBstats import __version__
    from PBstats.diagnostics.quality import _SEVERITY_ORDER, diagnose_signal
    from PBstats.features import functions as F
    from PBstats.visualization import plots

    multi = d.data.ndim == 2
    sub = d.channel(channel) if multi else d
    x = sub.data
    fs = sub.fs
    label = d.label or "signal"
    title = title or f"Signal report — {label}"

    rep = diagnose_signal(x, fs, sub.label, signal_type=signal_type)
    g = rep.grade
    color = {"A": "var(--ok)", "B": "var(--ok)", "C": "var(--warn)", "D": "var(--bad)", "F": "var(--bad)"}[g]

    chips = [f"{len(x):,} samples"]
    if fs:
        chips += [f"{fs:g} Hz", f"{len(x) / fs:,.1f} s"]
    if multi:
        chips.append(f"{d.n_channels} channels — showing '{d.channel_names[channel] if isinstance(channel, int) else channel}'")
    if signal_type:
        chips.append(f"type: {signal_type.upper()}")
    if d.annotations is not None and len(d.annotations):
        chips.append(f"{len(d.annotations):,} annotations")

    parts = [f"<header><h1>{html.escape(title)}</h1><p>Generated {_dt.datetime.now():%Y-%m-%d %H:%M} "
             f"with PBstats {__version__}</p><div class='chips'>"
             + "".join(f"<span class='chip'>{html.escape(c)}</span>" for c in chips) + "</div></header>"]

    # ── Quality ───────────────────────────────────────────────────────────
    issues = sorted(rep.issues, key=lambda i: _SEVERITY_ORDER[i.severity])
    li = "".join(f"<li><span class='pill {i.severity}'>{i.severity}</span>{html.escape(i.message)}</li>"
                 for i in issues) or "<li>No issues detected.</li>"
    rx = ""
    if rep.fixes:
        rx = ("<h2 style='margin-top:18px'>Prescribed cleaning</h2><pre>"
              + html.escape("\n".join(f"{n}({', '.join(f'{k}={v!r}' for k, v in kw.items())})"
                                      for n, kw in rep.fixes))
              + "</pre><p class='muted'>Apply with <code>d.auto_clean()</code> or reuse as a pipeline:</p><pre>"
              + html.escape(json.dumps(rep.pipeline.to_dict(), indent=2)) + "</pre>")
    parts.append(
        f"<section><h2>Signal quality</h2><div class='score'><div class='ring' "
        f"style='border:7px solid {color}'><div>{rep.score}<small>grade {g}</small></div></div>"
        f"<ul class='issues'>{li}</ul></div>{rx}</section>")

    parts.append("<section><h2>Diagnosis view</h2>"
                 + _fig_to_img(lambda buf: plots.plot_quality(rep, save_path=buf)) + "</section>")

    # Quality over time for long recordings
    if fs and len(x) / fs >= 60:
        from PBstats.core.data import Data
        tl = Data(x, fs=fs, label=sub.label).quality_timeline(window=10.0, signal_type=signal_type)
        bad = tl[~tl["ok"]]
        note = (f"{len(bad)} of {len(tl)} ten-second windows have problems"
                + (f" (worst at {bad.loc[bad.score.idxmin(), 't_start']:.0f} s)." if len(bad) else "."))
        parts.append("<section><h2>Quality over time</h2><p class='muted'>" + html.escape(note) + "</p>"
                     + _fig_to_img(lambda buf: plots.plot_quality_timeline(tl, sub.label, save_path=buf))
                     + "</section>")

    # ── Statistics + features ─────────────────────────────────────────────
    finite = x[np.isfinite(x)]
    stats_rows = []
    if len(finite) > 2:
        from scipy import stats as sp
        q25, med, q75 = np.percentile(finite, [25, 50, 75])
        stats_rows = [("Mean", finite.mean()), ("Std", finite.std(ddof=1)), ("Median", med),
                      ("IQR", q75 - q25), ("Min", finite.min()), ("Max", finite.max()),
                      ("RMS", np.sqrt(np.mean(finite ** 2))), ("Skewness", sp.skew(finite)),
                      ("Kurtosis", sp.kurtosis(finite))]
        for k in ("ecg_hr_bpm", "ecg_template_corr", "mains_db", "drift_fraction", "spike_count"):
            if k in rep.metrics:
                stats_rows.append((k.replace("_", " "), rep.metrics[k]))

    feat_html = ""
    if features and len(finite) == len(x) and len(x) >= 64:
        seg = x
        note = ""
        if fs and len(x) / fs > max_feature_seconds:
            seg = x[: int(max_feature_seconds * fs)]
            note = f" (first {max_feature_seconds:g} s)"
        include = ["time", "freq", "nonlinear"] if fs else ["time", "nonlinear"]
        if signal_type == "ecg" and fs:
            include.append("hrv")
        try:
            feats = F.extract(seg, fs, include)
            items = list(feats.items())
            half = (len(items) + 1) // 2
            feat_html = (f"<section><h2>Features{html.escape(note)}</h2><div class='grid'>"
                         + _table(items[:half], ("Feature", "Value"))
                         + _table(items[half:], ("Feature", "Value")) + "</div></section>")
        except Exception as e:  # report must never fail on one feature group
            feat_html = f"<section><h2>Features</h2><p class='muted'>Not computed: {html.escape(str(e))}</p></section>"
    elif features:
        feat_html = ("<section><h2>Features</h2><p class='muted'>Skipped — the signal contains "
                     "missing values (run remove_missing() first).</p></section>")

    figs = [lambda buf: plots.plot_signal(x, fs=fs or 1.0, label=sub.label or label, save_path=buf)]
    if fs and len(finite) == len(x):
        from PBstats.core.data import Data
        p = Data(x, fs=fs, label=sub.label).psd().psd_result
        figs.append(lambda buf: plots.plot_psd(p, save_path=buf))
    parts.append("<section><h2>Signal</h2>" + "".join(_fig_to_img(f) for f in figs) + "</section>")
    if stats_rows:
        parts.append("<section><h2>Statistics</h2>" + _table(stats_rows) + "</section>")
    parts.append(feat_html)

    # ── Provenance ────────────────────────────────────────────────────────
    if d.steps:
        try:
            pipe_json = json.dumps(d.to_pipeline(f"{label}_session").to_dict(), indent=2)
        except TypeError:
            pipe_json = "\n".join(f"{n}({kw})" for n, kw in d.steps)
        parts.append("<section><h2>Processing history</h2><p class='muted'>Everything applied to this "
                     "recording, as a replayable pipeline (<code>Pipeline.from_dict</code>):</p><pre>"
                     + html.escape(pipe_json) + "</pre></section>")

    parts.append(f"<footer>PBstats {__version__} · diagnosis thresholds are heuristics — "
                 f"see the documentation for their validation.</footer>")
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body><main>"
           + "".join(parts) + "</main></body></html>")
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
    return doc


class ReportMixin:

    def report(
        self,
        path: Optional[str] = None,
        title: Optional[str] = None,
        signal_type: Optional[str] = None,
        channel=0,
        features: bool = True,
    ) -> str:
        """
        Write a self-contained HTML report (figures embedded) and return its
        HTML. Contents: quality score, issues and prescribed cleaning,
        diagnosis figure, quality-over-time strip (recordings ≥ 60 s),
        signal + PSD, statistics, features and the processing history.

        signal_type : 'ecg' adds heartbeat checks and HRV features
        channel     : which channel to detail for multichannel data
        """
        return build_report(self, path, title, signal_type, channel, features)
