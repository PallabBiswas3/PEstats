from __future__ import annotations

import copy as _copy
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from PBstats.core.utils import safe_print

PIPELINE_FORMAT_VERSION = 1


def _jsonable(value: Any, where: str) -> Any:
    """Convert a step argument into a JSON-safe value, or fail loudly."""
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, where) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v, where) for k, v in value.items()}
    raise TypeError(
        f"Step '{where}' has an argument of type {type(value).__name__} that cannot be "
        f"saved to JSON (arrays/objects are data, not configuration)."
    )


@dataclass
class PipelineStep:
    """One step — a Data method name + its kwargs."""
    name: str
    kwargs: dict = field(default_factory=dict)
    duration_ms: float = 0.0


class Pipeline:
    """
    Define a preprocessing + analysis workflow once; apply it to any number
    of ``Data`` objects reproducibly.

    Usage
    -----
    pipe = (Pipeline("ecg_clean")
            .add("remove_missing", strategy="interpolate")
            .add("bandpass", low=0.5, high=40)
            .add("notch", freq=50)
            .add("standardize"))

    clean = pipe.run(Data(signal, fs=500))
    pipe.save("ecg_pipeline.json")
    same  = Pipeline.load("ecg_pipeline.json")

    Tip: explore interactively on one recording, then call
    ``d.to_pipeline()`` to turn what you did into a Pipeline.
    """

    def __init__(self, name: str = "pipeline"):
        self.name = name
        self.steps: list[PipelineStep] = []
        self._run_log: list[dict] = []

    # ── building ──────────────────────────────────────────────────────────

    def add(self, name: str, **kwargs) -> "Pipeline":
        """Add a step by ``Data`` method name, e.g. add("notch", freq=50)."""
        self.steps.append(PipelineStep(name=name, kwargs=dict(kwargs)))
        return self

    def __add__(self, other: "Pipeline") -> "Pipeline":
        """Concatenate two pipelines: clean + features."""
        out = _copy.deepcopy(self)
        out.name = f"{self.name}+{other.name}"
        out._run_log = []
        out.steps.extend(_copy.deepcopy(other.steps))
        if isinstance(other, FunctionalPipeline):
            fp = FunctionalPipeline(out.name)
            fp.steps = out.steps
            fp._fn_registry = {**getattr(self, "_fn_registry", {}), **other._fn_registry}
            return fp
        return out

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __repr__(self) -> str:
        return f"Pipeline('{self.name}', steps=[{' → '.join(s.name for s in self.steps)}])"

    # ── execution ─────────────────────────────────────────────────────────

    def _resolve(self, data_obj, step: PipelineStep) -> Callable[[], Any]:
        method = getattr(data_obj, step.name)
        return lambda: method(**step.kwargs)

    def _is_valid(self, data_obj, name: str) -> bool:
        # Private names are refused: pipelines may be loaded from untrusted JSON.
        return (not name.startswith("_")
                and callable(getattr(data_obj, name, None)))

    def _validate_steps(self, data_obj) -> None:
        missing = [s.name for s in self.steps if not self._is_valid(data_obj, s.name)]
        if missing:
            public = sorted(m for m in dir(data_obj)
                            if not m.startswith("_") and callable(getattr(data_obj, m, None)))
            raise AttributeError(
                f"Pipeline '{self.name}' has steps not found on Data: {missing}\n"
                f"Available methods: {public}"
            )

    def run(self, data_obj, verbose: bool = False, copy: bool = False) -> Any:
        """
        Execute all steps in order.

        Parameters
        ----------
        data_obj : a PBstats.Data instance
        verbose  : print each step with its duration
        copy     : run on a copy, leaving ``data_obj`` untouched

        Returns
        -------
        The processed Data object.
        """
        if copy:
            data_obj = data_obj.copy()
        self._validate_steps(data_obj)

        record = {"pipeline": self.name, "label": getattr(data_obj, "label", ""),
                  "steps": [], "total_ms": 0.0}
        for step in self.steps:
            call = self._resolve(data_obj, step)
            t0 = time.perf_counter()
            call()
            step.duration_ms = (time.perf_counter() - t0) * 1000
            if verbose:
                safe_print(f"  [{step.duration_ms:8.2f} ms]  {step.name}({step.kwargs or ''})")
            record["steps"].append({"step": step.name, "kwargs": step.kwargs,
                                    "duration_ms": round(step.duration_ms, 3)})
        record["total_ms"] = sum(s["duration_ms"] for s in record["steps"])
        self._run_log.append(record)
        return data_obj

    __call__ = run

    def run_batch(
        self,
        data_list: list,
        verbose: bool = False,
        on_error: str = "raise",
        copy: bool = False,
    ) -> list:
        """
        Run the pipeline on many Data objects.

        on_error : 'raise' (default) — stop at the first error
                   'warn'  — print a warning and continue
                   'skip'  — continue silently
        Failed items are excluded from the result; details are kept in
        ``self.batch_errors``.
        """
        if on_error not in ("raise", "warn", "skip"):
            raise ValueError("on_error must be 'raise', 'warn' or 'skip'")
        results, self.batch_errors = [], []
        for i, d in enumerate(data_list):
            label = getattr(d, "label", "") or f"sample_{i}"
            try:
                if verbose:
                    safe_print(f"\nProcessing [{i + 1}/{len(data_list)}]: {label}")
                results.append(self.run(d, verbose=verbose, copy=copy))
            except Exception as e:
                msg = f"Error on '{label}': {type(e).__name__}: {e}"
                self.batch_errors.append({"index": i, "label": label, "error": msg})
                if on_error == "raise":
                    raise
                if on_error == "warn":
                    safe_print(f"  WARNING — {msg}")
        if self.batch_errors and on_error == "warn":
            safe_print(f"\nBatch complete: {len(results)} succeeded, {len(self.batch_errors)} failed.")
        return results

    # ── timing ────────────────────────────────────────────────────────────

    def timing_report(self) -> dict:
        """Per-step mean/max duration across all runs — find bottlenecks."""
        times: dict[str, list[float]] = {}
        for run in self._run_log:
            for s in run["steps"]:
                times.setdefault(s["step"], []).append(s["duration_ms"])
        return {
            name: {"mean_ms": round(float(np.mean(t)), 3),
                   "max_ms": round(float(np.max(t)), 3),
                   "n_runs": len(t)}
            for name, t in times.items()
        }

    # ── persistence ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "format_version": PIPELINE_FORMAT_VERSION,
            "name": self.name,
            "steps": [{"name": s.name, "kwargs": _jsonable(s.kwargs, s.name)}
                      for s in self.steps],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Pipeline":
        pipe = cls(name=payload.get("name", "pipeline"))
        for s in payload.get("steps", []):
            pipe.add(s["name"], **s.get("kwargs", {}))
        return pipe

    def save(self, path: str) -> None:
        """Save the step definitions (not data) to JSON."""
        payload = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Pipeline":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ── introspection ─────────────────────────────────────────────────────

    def summary(self) -> None:
        safe_print(f"\nPipeline: '{self.name}'  ({len(self.steps)} steps)")
        safe_print("─" * 48)
        for i, s in enumerate(self.steps, 1):
            kw = ", ".join(f"{k}={v!r}" for k, v in s.kwargs.items())
            safe_print(f"  {i:2d}. {s.name}({kw})")
        safe_print("─" * 48)


class FunctionalPipeline(Pipeline):
    """
    Pipeline that also accepts arbitrary callables ``fn(data_obj)`` as steps,
    for custom transforms that aren't Data methods.

    Example
    -------
    def scale_by_10(d):
        d.data = d.data * 10
        d._record("scale_by_10")
        return d

    pipe = (FunctionalPipeline("custom")
            .add("remove_dc")
            .add_fn(scale_by_10)
            .add("standardize"))

    Note: custom callables cannot be saved to JSON.
    """

    def __init__(self, name: str = "pipeline"):
        super().__init__(name)
        self._fn_registry: dict[str, Callable] = {}

    def add_fn(self, fn: Callable, name: str | None = None, **kwargs) -> "FunctionalPipeline":
        """Add a callable step; extra kwargs are passed as fn(data_obj, **kwargs)."""
        step_name = name or fn.__name__
        self._fn_registry[step_name] = fn
        self.steps.append(PipelineStep(name=step_name, kwargs=dict(kwargs)))
        return self

    def _resolve(self, data_obj, step: PipelineStep):
        if step.name in self._fn_registry:
            fn = self._fn_registry[step.name]
            return lambda: fn(data_obj, **step.kwargs)
        return super()._resolve(data_obj, step)

    def _is_valid(self, data_obj, name: str) -> bool:
        return name in self._fn_registry or super()._is_valid(data_obj, name)

    def to_dict(self) -> dict:
        custom = [s.name for s in self.steps if s.name in self._fn_registry]
        if custom:
            raise ValueError(
                f"Cannot save custom callable steps {custom} to JSON. "
                f"Make them Data methods, or rebuild them with add_fn() after loading."
            )
        return super().to_dict()
