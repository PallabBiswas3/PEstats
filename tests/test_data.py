import numpy as np
import pytest
from PBstats import Data

def test_preprocessing_chain():
    """Test a standard sequence of cleaning and normalization."""
    raw = [1.0, 2.0, np.nan, 100.0, 3.0]
    d = (Data(raw)
         .remove_missing(strategy="interpolate")
         .remove_outliers(method="zscore", threshold=2.0)
         .minmax())
    
    # History should record 3 operations
    assert len(d.history()) == 3
    # Max value should be 1.0 due to minmax
    assert np.isclose(np.max(d.data), 1.0)
    # No NaNs should remain
    assert not np.isnan(d.data).any()

def test_vectorized_imputation_2d():
    """Verify the new optimized 2D ffill/bfill logic."""
    raw = np.array([
        [1.0, np.nan],
        [np.nan, 2.0],
        [3.0, np.nan],
        [np.nan, 4.0]
    ])
    d = Data(raw)
    
    # Run ffill
    d.impute(method="ffill")
    # Col 0: [1, 1, 3, 3]
    # Col 1: [nan, 2, 2, 4]
    assert d.data[1, 0] == 1.0
    assert d.data[3, 0] == 3.0
    assert d.data[2, 1] == 2.0
    assert np.isnan(d.data[0, 1]) # First element is still NaN
    
    # Run bfill to catch the first element
    d.impute(method="bfill")
    assert d.data[0, 1] == 2.0

def test_normalization_edge_cases():
    """Ensure zero-variance signals don't cause crashes."""
    # Constant signal
    d = Data([1.0, 1.0, 1.0])
    
    # Standardize would divide by std=0
    d.standardize()
    assert np.all(d.data == 0)
    
    # Minmax would divide by max-min=0
    d.reset().minmax()
    assert np.all(d.data == 0)

def test_data_coercion():
    """Verify that Data class correctly handles different input types."""
    import pandas as pd
    
    # Test list
    assert isinstance(Data([1, 2, 3]).data, np.ndarray)
    
    # Test Series
    s = pd.Series([1.0, 2.0, 3.0], name="test")
    d = Data(s)
    assert d.label == "" # unless provided
    assert d.data.shape == (3,)
    
    # Test DataFrame
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    d = Data(df)
    assert d.data.shape == (2, 2)
