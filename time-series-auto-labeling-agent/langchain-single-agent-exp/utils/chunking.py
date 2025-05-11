"""
Utility functions for chunking time series data.
"""
from typing import List, Any

def chunk_time_series(
    time_series: List[Any], chunk_size: int = 50, overlap: int = 10
) -> List[List[Any]]:
    """
    Split a time series into overlapping chunks.
    
    Args:
        time_series: List of values representing the time series.
        chunk_size: Size of each chunk.
        overlap: Overlap between consecutive chunks.
        
    Returns:
        List of chunked time series data.
    """
    # If time series is shorter than chunk size, return it as is
    if len(time_series) <= chunk_size:
        return [time_series]
    
    # Calculate step size
    step = chunk_size - overlap
    
    # Generate chunks
    chunks = []
    for i in range(0, len(time_series) - chunk_size + 1, step):
        chunk = time_series[i:i + chunk_size]
        chunks.append(chunk)
    
    # Add the last chunk if it doesn't align perfectly
    if (len(time_series) - chunk_size) % step != 0:
        chunks.append(time_series[-chunk_size:])
    
    return chunks 