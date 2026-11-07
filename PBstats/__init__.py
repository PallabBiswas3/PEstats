"""
PBstats — signal-aware analysis, from raw recording to ML-ready features.

    from PBstats import Data
    d = Data(raw, fs=500, label="ecg").diagnose(verbose=True)   # what's wrong?
    d.auto_clean().auto_features()                              # fix it, featurise it
    d.to_pipeline("ecg_clean").save("ecg_clean.json")           # make it reproducible
"""
from PBstats.core.data import Data
from PBstats.core.pipeline import FunctionalPipeline, Pipeline
from PBstats.augment import Augmenter, make_windows, to_torch_dataset
from PBstats.diagnostics.quality import QualityReport
from PBstats.epochs import Epochs

__version__ = "0.3.0"
__all__ = [
    "Data", "Pipeline", "FunctionalPipeline", "QualityReport", "Epochs",
    "Augmenter", "make_windows", "to_torch_dataset",
    "SignalFeatureExtractor", "extract_features",
]


def __getattr__(name):
    # Lazy: importing scikit-learn is only paid for when the ML bridge is used.
    if name in ("SignalFeatureExtractor", "extract_features"):
        from PBstats import ml
        return getattr(ml, name)
    raise AttributeError(f"module 'PBstats' has no attribute {name!r}")
