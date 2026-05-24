import numpy as np
import pytest
import os
import json
from PBstats import Data, Pipeline, FunctionalPipeline

@pytest.fixture
def ecg_generator():
    fs = 1000
    def _make(seed=0):
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 1, fs)
        sig = np.sin(2 * np.pi * 50 * t) + 0.2 * rng.standard_normal(fs)
        sig[100:105] = np.nan
        return sig, fs
    return _make

def test_pipeline_execution(ecg_generator):
    sig, fs = ecg_generator()
    pipe = (
        Pipeline("test_pipe")
        .add("remove_missing")
        .add("remove_dc")
        .add("standardize")
    )
    
    d = pipe.run(Data(sig, fs=fs))
    
    assert len(d.history()) == 3
    assert not np.isnan(d.data).any()
    assert np.isclose(np.mean(d.data), 0, atol=1e-10)

def test_pipeline_persistence(tmp_path, ecg_generator):
    """Test saving and loading pipelines from JSON."""
    pipe_path = tmp_path / "test_pipeline.json"
    
    pipe = (
        Pipeline("persistence_test")
        .add("remove_missing", strategy="zero")
        .add("minmax")
    )
    
    pipe.save(str(pipe_path))
    assert os.path.exists(pipe_path)
    
    # Load and verify
    reloaded = Pipeline.load(str(pipe_path))
    assert reloaded.name == "persistence_test"
    assert len(reloaded.steps) == 2
    assert reloaded.steps[0].kwargs["strategy"] == "zero"

def test_functional_pipeline(ecg_generator):
    """Test pipeline with custom function steps."""
    sig, fs = ecg_generator()
    
    def my_custom_step(d_obj):
        d_obj.data = d_obj.data * 10
        d_obj._record("multiplied")
        return d_obj
    
    pipe = (
        FunctionalPipeline("custom")
        .add("remove_missing")
        .add_fn(my_custom_step, name="custom_mult")
    )
    
    d = pipe.run(Data(sig, fs=fs))
    assert "multiplied" in d.history()
    # verify logic
    assert np.nanmax(sig) * 10 == pytest.approx(np.max(d.data))

def test_pipeline_batch_processing(ecg_generator):
    samples = [Data(ecg_generator(i)[0], fs=1000) for i in range(5)]
    pipe = Pipeline("batch").add("remove_missing").add("standardize")
    
    results = pipe.run_batch(samples)
    assert len(results) == 5
    for r in results:
        assert not np.isnan(r.data).any()

def test_pipeline_invalid_step():
    pipe = Pipeline("invalid").add("this_method_does_not_exist")
    d = Data([1, 2, 3])
    
    with pytest.raises(AttributeError):
        pipe.run(d)
