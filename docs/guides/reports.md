# Reports

```python
d.report("patient_001.html", signal_type="ecg")
```

writes one self-contained HTML file — figures embedded, no external assets, light and dark themes — containing:

- the quality score, issues and prescribed cleaning (with the pipeline as JSON)
- the diagnosis figure (signal with problem regions highlighted, spectrum)
- a quality-over-time strip for recordings of 60 s or more
- the signal and its Welch PSD
- descriptive statistics and extracted features (HRV with `signal_type="ecg"`)
- the full processing history as a replayable pipeline

Reports are easy to email, attach to an issue, or archive next to the data. For multichannel data, `channel=` selects which channel to detail. Features are computed on the first 5 minutes of very long recordings to keep generation fast.
