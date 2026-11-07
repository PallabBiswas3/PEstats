# Events, epochs & forecasting

## Change points

Find where the signal's behaviour changes, with PELT exact segmentation (Killick et al., *JASA* 2012):

```python
d.detect_change_points(cost="meanvar")   # level and/or variability changes
r = d.change_point_result
r.times          # change points in seconds
r.segments       # start_s, end_s, n, mean, std per regime
d.plot_events()
```

- `cost="mean"` assumes constant variance (it will split variance changes into many level changes); `"meanvar"` handles both.
- `penalty` controls how many changes are found (default: BIC). `min_duration` sets the shortest segment.
- Recordings longer than 10,000 samples are segmented on a block grid with exact segment statistics, so a million samples take a few seconds.

## Anomalies

```python
res = d.anomaly_scores(window=5)          # Isolation Forest on windowed features
res.sort_values("score").tail()           # most unusual windows
d.plot_events()                           # anomalous windows shaded
```

## Epochs

Cut fixed-length segments around events and average them — the event-related-potential workflow:

```python
ep = d.epochs("stimulus", tmin=-0.2, tmax=0.8, baseline=(None, 0), reject=150)
erp = ep.average()          # a Data object
ep.plot()
X = ep.to_numpy()           # (epochs, channels, times) float32 for deep learning
```

`events` can be times in seconds, a DataFrame with `onset`/`label`, an annotation label or list of labels, or `None` for every annotation. Epochs that run past the recording edges or exceed `reject` peak-to-peak are dropped (`ep.n_dropped`).

## Forecasting

```python
d.forecast(horizon=10.0)                 # 10 s ahead
d.forecast(steps=100, holdout=100)       # honest check on held-out data
d.forecast_result.metrics                # MAE, RMSE, coverage, skill vs naive
d.plot_forecast()
```

By default the ARIMA order and trend term are chosen by **rolling-origin validation**: each candidate model forecasts the last two stretches of the training data, and the most accurate is refit on everything. In testing, in-sample AIC often picked models that forecast badly, so it is available but not the default (`select="aic"`). A refit model whose forecast diverges is replaced by the next-best candidate.

`metrics["skill"]` compares against simply repeating the last value: above 0 means the model adds something. ARIMA suits smooth, autocorrelated series (sensor trends, physiological rates, demand); it will not forecast ECG morphology — forecast derived series such as heart rate instead.
