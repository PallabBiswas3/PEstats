# Pipelines

A `Pipeline` is an ordered list of `Data` method calls with their arguments. It can be saved to JSON, reloaded and run on any number of recordings.

## Explore, then freeze

Every method you call on `Data` is recorded with its full arguments:

```python
d = (Data(x, fs=500)
     .remove_missing()
     .bandpass(0.5, 40)
     .notch(50)
     .normalize("zscore"))

d.steps
# [('remove_missing', {'strategy': 'interpolate'}),
#  ('bandpass', {'low': 0.5, 'high': 40, 'order': 4, 'zero_phase': True}), ...]

pipe = d.to_pipeline("ecg_clean")
pipe.save("ecg_clean.json")
```

Calls made *internally* (e.g. `normalize("zscore")` dispatching to `standardize()`) are not double-recorded, so replaying reproduces exactly what you did.

## Build explicitly

```python
from PBstats import Pipeline

pipe = (Pipeline("ecg_clean")
        .add("bandpass", low=0.5, high=40)
        .add("notch", freq=50)
        .add("standardize"))

clean = pipe.run(Data(x, fs=500))              # modifies in place
clean = pipe.run(Data(x, fs=500), copy=True)   # leaves the input untouched
results = pipe.run_batch(list_of_data, on_error="warn")
pipe.batch_errors                               # what failed and why
pipe.timing_report()                            # per-step timing
```

Pipelines compose: `cleaning + features`. `FunctionalPipeline.add_fn(fn)` adds arbitrary callables (these cannot be saved to JSON — PBstats refuses rather than silently dropping them).

## Safety

Loaded pipelines may come from someone else's JSON file, so only public `Data` methods can be called — names starting with `_` are refused.
