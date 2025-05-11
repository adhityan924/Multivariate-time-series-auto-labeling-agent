import numpy as np
from scipy.fft import fft
from typing import List

def compute_embedding(chunk: np.ndarray) -> List[float]:
    """
    Computes an embedding for a time series chunk using FFT
    and basic time-domain features.

    Args:
        chunk: A numpy array representing the time series chunk.

    Returns:
        A list of floats representing the computed embedding.
        Returns an empty list if the chunk is empty or invalid.
    """
    if chunk is None or chunk.size == 0:
        print("Warning: Received empty or invalid chunk for embedding.")
        return [] # Return empty list for invalid input

    # Ensure chunk is float type for calculations
    chunk = chunk.astype(float)

    # --- Time-Domain Features ---
    mean_val = np.mean(chunk)
    std_dev = np.std(chunk)
    min_val = np.min(chunk)
    max_val = np.max(chunk)
    time_domain_features = [mean_val, std_dev, min_val, max_val]

    # --- Frequency-Domain Features (FFT) ---
    # Compute the Fast Fourier Transform
    fft_vals = fft(chunk)
    # Get the magnitudes (absolute values) of the first half of the FFT result
    # (since the second half is redundant for real-valued inputs)
    fft_magnitudes = np.abs(fft_vals[:len(fft_vals)//2])

    # --- Combine Features ---
    # Ensure all features are floats before combining
    combined_features = time_domain_features + fft_magnitudes.tolist()

    # Handle potential NaN/inf values (replace with 0)
    # This can happen with std_dev on constant chunks, or potentially FFT issues
    combined_features_cleaned = [0.0 if np.isnan(f) or np.isinf(f) else float(f) for f in combined_features]

    return combined_features_cleaned
