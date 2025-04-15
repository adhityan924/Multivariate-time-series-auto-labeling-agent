# Evaluation Script for Metric C: Output Stability (Jaccard Index)

import argparse
import logging
import os
import sys
import itertools

# Add project root to sys.path to allow importing other evaluation modules if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Reuse the parsing function from calculate_similarity
# Note: Ensure calculate_similarity.py's parse_agent_output handles encoding errors
try:
    from evaluation.calculate_similarity import parse_agent_output
except ImportError:
    logging.error("Could not import 'parse_agent_output' from 'calculate_similarity.py'. Make sure it exists.")
    # Define a dummy function if import fails to avoid crashing later
    def parse_agent_output(log_filepath: str) -> list[dict]:
        logging.error(f"Using dummy parse_agent_output due to import error. Cannot parse {log_filepath}")
        return []


# Setup basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

def segments_to_set(segment_list: list[dict]) -> set[tuple[int, int]]:
    """Converts a list of segment dictionaries to a set of tuples for comparison."""
    segment_set = set()
    for segment in segment_list:
        try:
            start = segment['start_row']
            end = segment['end_row']
            segment_set.add((start, end))
        except KeyError:
            logging.warning(f"Segment dictionary missing 'start_row' or 'end_row': {segment}")
    return segment_set

def are_segments_similar(seg1: tuple[int, int], seg2: tuple[int, int], tolerance: int = 50) -> bool:
    """
    Determines if two segments are similar based on their start and end points.

    Args:
        seg1: First segment as (start_row, end_row)
        seg2: Second segment as (start_row, end_row)
        tolerance: Maximum allowed difference in start/end points

    Returns:
        True if segments are similar, False otherwise
    """
    start1, end1 = seg1
    start2, end2 = seg2

    # Check if start and end points are within tolerance
    start_diff = abs(start1 - start2)
    end_diff = abs(end1 - end2)

    # Segments are similar if both start and end points are within tolerance
    return start_diff <= tolerance and end_diff <= tolerance

def calculate_jaccard_index(set1: set, set2: set) -> float:
    """Calculates the standard Jaccard index between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union == 0:
        return 1.0 # Define Jaccard index as 1 if both sets are empty
    else:
        return intersection / union

def calculate_approximate_jaccard_index(segments1: set[tuple[int, int]], segments2: set[tuple[int, int]],
                                      tolerance: int = 50) -> float:
    """
    Calculates an approximate Jaccard index between two sets of segments,
    considering segments as similar if their start and end points are within tolerance.

    Args:
        segments1: First set of segments as (start_row, end_row) tuples
        segments2: Second set of segments as (start_row, end_row) tuples
        tolerance: Maximum allowed difference in start/end points

    Returns:
        Approximate Jaccard index
    """
    logging.info(f"Calculating approximate Jaccard index with tolerance {tolerance}")
    logging.info(f"  Set 1: {segments1}")
    logging.info(f"  Set 2: {segments2}")

    if not segments1 and not segments2:
        return 1.0  # Both sets are empty, perfect match

    if not segments1 or not segments2:
        return 0.0  # One set is empty, no match

    # Count similar segments (approximate intersection)
    similar_count = 0
    matched_segments2 = set()  # Keep track of segments in set2 that have been matched

    # For each segment in set1, check if there's a similar segment in set2
    for seg1 in segments1:
        for seg2 in segments2:
            if seg2 in matched_segments2:
                continue  # Skip segments that have already been matched

            if are_segments_similar(seg1, seg2, tolerance):
                similar_count += 1
                matched_segments2.add(seg2)
                logging.info(f"  Found similar segments: {seg1} and {seg2}")
                break  # Found a match for this segment, move to the next one

    # Calculate approximate Jaccard index
    # The union is the total number of unique segments
    union = len(segments1) + len(segments2) - similar_count

    result = similar_count / union if union > 0 else 0.0
    logging.info(f"  Similar count: {similar_count}, Union: {union}")
    logging.info(f"  Approximate Jaccard index: {result:.4f}")

    return result

def main():
    parser = argparse.ArgumentParser(description="Calculate Output Stability (Jaccard Index) across multiple runs.")
    parser.add_argument("log_filepaths", nargs='+', help="Paths to the agent's log files from multiple runs.")
    parser.add_argument("--tolerance", type=int, default=50, help="Tolerance for approximate matching (default: 50 rows)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if len(args.log_filepaths) < 2:
        logging.error("Need at least two log files to calculate stability.")
        sys.exit(1)

    logging.info(f"Processing {len(args.log_filepaths)} log files for stability analysis...")

    # 1. Parse segment lists from all log files
    all_segment_sets = []
    for log_path in args.log_filepaths:
        if not os.path.exists(log_path):
            logging.warning(f"Log file not found: {log_path}. Skipping.")
            continue
        logging.info(f"Parsing segments from: {log_path}")
        segments = parse_agent_output(log_path)
        logging.info(f"  Raw segments: {segments}")
        segment_set = segments_to_set(segments)
        logging.info(f"  Segment set: {segment_set}")
        all_segment_sets.append(segment_set)
        logging.info(f"  Found {len(segment_set)} unique segments.")

    if len(all_segment_sets) < 2:
        logging.error("Could not successfully parse segments from at least two log files. Cannot calculate stability.")
        sys.exit(1)

    # 2. Calculate pairwise Jaccard indices (both standard and approximate)
    standard_jaccard_indices = []
    approximate_jaccard_indices = []

    # Use the tolerance from command-line arguments
    tolerance = args.tolerance  # Consider segments similar if start/end points are within this many rows
    logging.info(f"Using tolerance of {tolerance} rows for approximate matching")

    for set1, set2 in itertools.combinations(all_segment_sets, 2):
        # Calculate standard Jaccard index (exact matching)
        std_index = calculate_jaccard_index(set1, set2)
        standard_jaccard_indices.append(std_index)

        # Calculate approximate Jaccard index (similarity-based matching)
        approx_index = calculate_approximate_jaccard_index(set1, set2, tolerance)
        approximate_jaccard_indices.append(approx_index)

        logging.info(f"Comparison between sets: {set1} and {set2}")
        logging.info(f"  Standard Jaccard Index: {std_index:.4f}")
        logging.info(f"  Approximate Jaccard Index: {approx_index:.4f}")

    # 3. Calculate average Jaccard indices
    avg_std_jaccard = sum(standard_jaccard_indices) / len(standard_jaccard_indices) if standard_jaccard_indices else 0
    avg_approx_jaccard = sum(approximate_jaccard_indices) / len(approximate_jaccard_indices) if approximate_jaccard_indices else 0

    # 4. Output results
    print("\n--- Output Stability Metrics (Metric C) ---")
    print(f"Number of runs compared: {len(all_segment_sets)}")
    print(f"Average Pairwise Jaccard Index (Exact Matching): {avg_std_jaccard:.4f}")
    print(f"Average Pairwise Jaccard Index (Approximate Matching): {avg_approx_jaccard:.4f}")
    print(f"Tolerance for approximate matching: {tolerance} rows")
    print("------------------------------------------")
    logging.info(f"Individual Standard Jaccard Indices: {standard_jaccard_indices}")
    logging.info(f"Individual Approximate Jaccard Indices: {approximate_jaccard_indices}")


if __name__ == "__main__":
    main()
