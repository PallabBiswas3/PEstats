import numpy as np
import pytest
from PBstats import Data

@pytest.fixture
def mixed_signal():
    """Fixture to provide a contaminated 50Hz signal."""
    rng = np.random.default_rng(0)
    fs = 1000
    t = np.linspace(0, 2, 2 * fs)
    signal = (
        np.sin(2 * np.pi * 50 * t)           # 50 Hz component
        + 0.5 * np.sin(2 * np.pi * 120 * t)  # 120 Hz harmonic
        + 0.1 * rng.standard_normal(len(t))  # noise
        + 2.0                                # DC offset
    )
    return signal, fs

def test_fft_analysis(mixed_signal):
    sig, fs = mixed_signal
    d = (Data(sig, fs=fs)
         .remove_dc()
         .fft(window="hann"))
    
    r = d.fft_result
    # Peak should be exactly 50Hz
    assert np.isclose(r.peak_frequency(), 50.0, atol=0.5)
    # 50Hz should have higher power than 120Hz
    assert r.band_power(45, 55) > r.band_power(115, 125)
    # Check shape of result
    assert len(r.freqs) == len(sig) // 2 + 1

def test_hilbert_envelope(mixed_signal):
    sig, fs = mixed_signal
    # Create a pulsed signal for better envelope testing
    sig_pulsed = sig * (np.sin(2 * np.pi * 0.5 * np.arange(len(sig))/fs) > 0)
    
    d = Data(sig_pulsed, fs=fs).hilbert()
    h = d.hilbert_result
    
    # Envelope should always be positive
    assert np.all(h.envelope >= 0)
    # Envelope should capture the signal peaks
    assert h.peak_envelope() >= np.max(sig_pulsed) * 0.9

def test_wavelet_transform(mixed_signal):
    sig, fs = mixed_signal
    # Use small subset for speed in tests
    d = Data(sig[:200], fs=fs).wavelet(n_scales=8)
    w = d.wavelet_result
    
    assert w.scalogram().shape == (8, 200)
    assert len(w.freqs) == 8

def test_stft_transform(mixed_signal):
    sig, fs = mixed_signal
    d = Data(sig, fs=fs).stft(nperseg=256)
    s = d.stft_result
    
    # Frequency bins should go up to Nyquist (fs/2)
    assert np.isclose(s.freqs[-1], fs/2)
    # Check matrix dimensions
    assert s.Zxx.ndim == 2
