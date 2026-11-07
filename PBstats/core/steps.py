"""
Step recording — the mechanism behind ``Data.to_pipeline()``.

Every chainable method on ``Data`` is wrapped with :func:`step`. When the user
calls it directly, the method name and its fully-bound arguments are stored in
``Data._steps``. Calls made *internally* by another step (e.g. ``normalize()``
dispatching to ``minmax()``) are not recorded, so replaying the recorded steps
reproduces exactly what the user did — no more, no less.
"""
from __future__ import annotations

import functools
import inspect
from typing import Callable


def step(fn: Callable) -> Callable:
    """Decorator marking a ``Data`` method as a replayable pipeline step."""
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        params = {k: v for k, v in bound.arguments.items() if k != "self"}

        depth = getattr(self, "_step_depth", 0)
        self._step_depth = depth + 1
        try:
            result = fn(self, *args, **kwargs)
        finally:
            self._step_depth = depth

        if depth == 0 and hasattr(self, "_steps"):
            self._steps.append((fn.__name__, params))
        return result

    wrapper._is_step = True
    return wrapper
