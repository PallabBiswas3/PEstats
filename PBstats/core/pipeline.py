from __future__ import annotations
import json
import time
import numpy as np
from typing import Callable, Any
from dataclasses import dataclass, field


@dataclass
class PipelineStep:
    """One step in the pipeline — a method name + its kwargs."""
    name: str                        # e.g. "remove_dc"
    kwargs: dict = field(default_factory=dict)
    duration_ms: float = 0.0        # filled after execution
    records: str = ""               # filled from Data._log


class Pipeline:
    """
    Define a preprocessing + transform workflow once.
    Apply it to any number of Data objects reproducibly.

    Usage
    -----
    pipe = (
        Pipeline("ecg_clean")
        .add("remove_missing", strategy="interpolate")
        .add("remove_dc")
        .add("remove_outliers", method="mad", threshold=3.0)
        .add("standardize")
        .add("hilbert")
    )

    result = pipe.run(Data(signal, fs=1000))
    result2 = pipe.run(Data(new_signal, fs=1000))   # same steps, new data

    pipe.save("ecg_pipeline.json")
    pipe2 = Pipeline.load("ecg_pipeline.json")
    """

    def __init__(self, name: str = "pipeline"):
        self.name  = name
        self.steps: list[PipelineStep] = []
        self._run_log: list[dict] = []     # history of all .run() calls

    # ── building ────────────────────────────────────────────────────────────

    def add(self, name: str, **kwargs) -> "Pipeline":
        """
        Add a step by method name.
        The name must match a method on the Data class exactly.

        Example
        -------
        pipe.add("remove_outliers", method="mad", threshold=3.5)
        """
        self.steps.append(PipelineStep(name=name, kwargs=kwargs))
        return self

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        step_names = " → ".join(s.name for s in self.steps)
        return f"Pipeline('{self.name}', steps=[{step_names}])"

    # ── execution ───────────────────────────────────────────────────────────

    def run(self, data_obj, verbose: bool = False) -> Any:
        """
        Execute all steps on a Data object in order.

        Parameters
        ----------
        data_obj : a pallabstats.Data instance
        verbose  : if True, print each step as it runs

        Returns
        -------
        The same Data object after all transforms applied.
        """
        self._validate_steps(data_obj)

        run_record = {
            "pipeline": self.name,
            "label": data_obj.label,
            "steps": [],
            "total_ms": 0.0,
        }

        for step in self.steps:
            method = getattr(data_obj, step.name)

            t_start = time.perf_counter()
            method(**step.kwargs)
            t_end   = time.perf_counter()

            step.duration_ms = (t_end - t_start) * 1000

            if verbose:
                print(f"  [{step.duration_ms:6.2f} ms]  {step.name}({step.kwargs})")

            run_record["steps"].append({
                "step":        step.name,
                "kwargs":      step.kwargs,
                "duration_ms": round(step.duration_ms, 3),
            })

        run_record["total_ms"] = sum(
            s["duration_ms"] for s in run_record["steps"]
        )
        self._run_log.append(run_record)

        return data_obj

    def _validate_steps(self, data_obj) -> None:
        """Check all step names exist on the Data class before running."""
        missing = [
            s.name for s in self.steps
            if not hasattr(data_obj, s.name)
        ]
        if missing:
            raise AttributeError(
                f"Pipeline '{self.name}' has steps not found on Data: {missing}\n"
                f"Check spelling — available methods: "
                f"{[m for m in dir(data_obj) if not m.startswith('_')]}"
            )

    # ── batch processing ─────────────────────────────────────────────────────

    def run_batch(
        self,
        data_list: list,
        verbose: bool = False,
        on_error: str = "raise",
    ) -> list:
        """
        Run the pipeline on a list of Data objects.

        Parameters
        ----------
        data_list : list of Data instances
        verbose   : print progress
        on_error  : 'raise' — stop on first error (default)
                    'skip'  — log error, continue with next sample
                    'warn'  — print warning, continue

        Returns
        -------
        List of processed Data objects (failed ones excluded if on_error != 'raise')
        """
        results = []
        errors  = []

        for i, d in enumerate(data_list):
            label = getattr(d, "label", f"sample_{i}")
            try:
                if verbose:
                    print(f"\nProcessing [{i+1}/{len(data_list)}]: {label}")
                results.append(self.run(d, verbose=verbose))

            except Exception as e:
                msg = f"Error on '{label}': {type(e).__name__}: {e}"
                errors.append({"label": label, "error": msg})

                if on_error == "raise":
                    raise
                elif on_error == "warn":
                    print(f"  WARNING — {msg}")
                elif on_error == "skip":
                    pass

        if errors and on_error != "raise":
            print(f"\nBatch complete: {len(results)} succeeded, "
                  f"{len(errors)} failed.")

        return results

    # ── timing report ────────────────────────────────────────────────────────

    def timing_report(self) -> dict:
        """
        Return per-step average timing across all .run() calls.
        Useful for finding bottlenecks in long pipelines.
        """
        if not self._run_log:
            return {}

        step_times: dict[str, list[float]] = {}
        for run in self._run_log:
            for s in run["steps"]:
                step_times.setdefault(s["step"], []).append(s["duration_ms"])

        return {
            step: {
                "mean_ms": round(np.mean(times), 3),
                "max_ms":  round(np.max(times), 3),
                "n_runs":  len(times),
            }
            for step, times in step_times.items()
        }

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save pipeline definition to JSON.
        This saves the step names and kwargs — not the data.
        Load it later with Pipeline.load(path).
        """
        payload = {
            "name":  self.name,
            "steps": [
                {"name": s.name, "kwargs": s.kwargs}
                for s in self.steps
            ],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Pipeline saved → {path}")

    @classmethod
    def load(cls, path: str) -> "Pipeline":
        """Load a pipeline from a saved JSON file."""
        with open(path) as f:
            payload = json.load(f)

        pipe = cls(name=payload["name"])
        for step in payload["steps"]:
            pipe.add(step["name"], **step["kwargs"])

        print(f"Pipeline loaded: {pipe}")
        return pipe

    # ── introspection ────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a readable summary of all pipeline steps."""
        print(f"\nPipeline: '{self.name}'  ({len(self.steps)} steps)")
        print("─" * 48)
        for i, step in enumerate(self.steps, 1):
            kwargs_str = ", ".join(f"{k}={v!r}" for k, v in step.kwargs.items())
            print(f"  {i:2d}. {step.name}({kwargs_str})")
        print("─" * 48)
class FunctionalPipeline(Pipeline):
    """
    Extended pipeline that supports arbitrary callables as steps.

    Useful when you need a custom transform that isn't a Data method yet.

    Example
    -------
    def my_bandpass(data_obj):
        from scipy.signal import butter, filtfilt
        b, a = butter(4, [40, 60], btype='band', fs=data_obj.fs)
        data_obj.data = filtfilt(b, a, data_obj.data)
        data_obj._record("custom_bandpass(40-60Hz)")
        return data_obj

    pipe = (
        FunctionalPipeline("custom")
        .add("remove_dc")
        .add_fn(my_bandpass, name="bandpass_40_60")
        .add("standardize")
    )
    """

    def __init__(self, name: str = "pipeline"):
        super().__init__(name)
        self._fn_registry: dict[str, Callable] = {}

    def add_fn(self, fn: Callable, name: str = None) -> "FunctionalPipeline":
        """Add an arbitrary callable as a pipeline step."""
        step_name = name or fn.__name__
        self._fn_registry[step_name] = fn
        self.steps.append(PipelineStep(name=step_name, kwargs={}))
        return self

    def run(self, data_obj, verbose: bool = False):
        self._validate_steps_fn(data_obj)

        for step in self.steps:
            t_start = time.perf_counter()

            if step.name in self._fn_registry:
                self._fn_registry[step.name](data_obj)
            else:
                getattr(data_obj, step.name)(**step.kwargs)

            step.duration_ms = (time.perf_counter() - t_start) * 1000
            if verbose:
                print(f"  [{step.duration_ms:6.2f} ms]  {step.name}")

        return data_obj

    def _validate_steps_fn(self, data_obj) -> None:
        missing = [
            s.name for s in self.steps
            if not hasattr(data_obj, s.name)
            and s.name not in self._fn_registry
        ]
        if missing:
            raise AttributeError(f"Steps not found: {missing}")