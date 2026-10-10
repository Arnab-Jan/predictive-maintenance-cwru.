import numpy as np
from scipy.stats import kurtosis, skew

def extract_time_features(window: np.ndarray) -> dict:
    """Calculates 9 time-domain statistical features from a 1D numpy window array."""
    abs_window = np.abs(window)
    max_val = float(np.max(window))
    min_val = float(np.min(window))
    mean_val = float(np.mean(window))
    std_val = float(np.std(window))
    rms = float(np.sqrt(np.mean(window**2)))
    
    eps = 1e-8
    mean_abs = float(np.mean(abs_window)) + eps
    peak_abs = float(np.max(abs_window))
    
    return {
        'max': max_val,
        'min': min_val,
        'mean': mean_val,
        'std': std_val,
        'rms': rms,
        'skewness': float(skew(window)),
        'kurtosis': float(kurtosis(window)),
        'crest_factor': peak_abs / (rms + eps),
        'form_factor': rms / mean_abs
    }

def extract_freq_features(window: np.ndarray, fs: int = 48000) -> dict:
    """Calculates 5 frequency-domain spectral features via Hann windowing + RFFT."""
    # Apply Hann window to minimize spectral leakage
    hann_window = np.hanning(len(window))
    windowed_sig = window * hann_window
    
    # Compute Real FFT
    fft_vals = np.abs(np.fft.rfft(windowed_sig))
    fft_freqs = np.fft.rfftfreq(len(window), d=1.0/fs)
    
    eps = 1e-8
    total_energy = np.sum(fft_vals) + eps
    
    # Spectral Centroid
    spectral_centroid = float(np.sum(fft_freqs * fft_vals) / total_energy)
    
    # Spectral Spread
    spectral_spread = float(np.sqrt(np.sum(((fft_freqs - spectral_centroid)**2) * fft_vals) / total_energy))
    
    # Peak Frequency
    peak_freq = float(fft_freqs[np.argmax(fft_vals)])
    
    # Low vs High Spectral Energy split at 12 kHz
    cutoff_idx = len(fft_vals) // 2
    low_band_energy = float(np.sum(fft_vals[:cutoff_idx]**2))
    high_band_energy = float(np.sum(fft_vals[cutoff_idx:]**2))
    
    return {
        'spectral_centroid': spectral_centroid,
        'spectral_spread': spectral_spread,
        'peak_freq': peak_freq,
        'low_band_energy': low_band_energy,
        'high_band_energy': high_band_energy
    }
