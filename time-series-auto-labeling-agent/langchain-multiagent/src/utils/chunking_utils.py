import numpy as np
from typing import List, Generator

def create_overlapping_chunks(data: np.ndarray, chunk_size: int, overlap: int) -> Generator[np.ndarray, None, None]:
    """
    Creates overlapping chunks from a 1D numpy array.

    Args:
        data: The 1D numpy array to chunk.
        chunk_size: The desired size of each chunk.
        overlap: The number of elements to overlap between consecutive chunks.

    Yields:
        Numpy arrays representing the overlapping chunks.
    """
    if data is None or data.ndim != 1 or data.size == 0:
        print("Warning: Invalid data provided for chunking. Must be a non-empty 1D numpy array.")
        return # Stop generation if data is invalid

    if chunk_size <= 0:
        print(f"Warning: Chunk size ({chunk_size}) must be positive. Skipping chunking.")
        return

    if overlap < 0 or overlap >= chunk_size:
        print(f"Warning: Overlap ({overlap}) must be non-negative and less than chunk size ({chunk_size}). Skipping chunking.")
        return

    step = chunk_size - overlap
    if step <= 0:
        print(f"Warning: Step size ({step}) based on chunk size ({chunk_size}) and overlap ({overlap}) must be positive. Skipping chunking.")
        return

    for i in range(0, len(data) - chunk_size + 1, step):
        chunk = data[i:i + chunk_size]
        yield chunk

    # Handle potential last partial chunk if needed, though typically we want full chunks
    # If the last full chunk doesn't reach the end, and we need to capture the tail:
    # last_full_chunk_end = (len(data) - chunk_size + 1 // step) * step + chunk_size
    # if last_full_chunk_end < len(data):
    #     # Option 1: Yield the last chunk starting from the last possible step
    #     # last_step_start = (len(data) - chunk_size + 1 // step) * step
    #     # yield data[last_step_start : last_step_start + chunk_size] # This might go out of bounds if not careful
    #
    #     # Option 2: Yield a chunk anchored to the end (might be smaller than chunk_size if padding isn't used)
    #     # start_index = max(0, len(data) - chunk_size)
    #     # yield data[start_index:]
    # For this implementation, we only yield full-sized chunks based on the step.
