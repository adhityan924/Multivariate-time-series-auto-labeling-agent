# Evaluation Script for Metric A: Similarity of Identified Segments

import argparse
import json
import logging
import os
import sys
import numpy as np
import pandas as pd
from tslearn.metrics import dtw
from scipy.spatial.distance import euclidean

# Add project root to sys.path to allow importing config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# import config # No longer importing config directly

# Setup basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_full_dataset(dataset_path: str) -> pd.DataFrame:
    """Loads the full time-series dataset."""
    try:
        df = pd.read_csv(dataset_path)
        logging.info(f"Successfully loaded dataset from {dataset_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"Dataset file not found at {dataset_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading dataset from {dataset_path}: {e}")
        return None

def parse_agent_output(log_filepath: str) -> list[dict]:
    """
    Parses the agent's log file to find and extract the 'identified_segments'
    list from the final JSON output block logged by run_experiment_1.py.
    """
    identified_segments = []
    json_string = ""
    found_json_start_marker = False
    parsing_json = False
    brace_count = 0

    try:
        # Add errors='ignore' to handle potential non-UTF8 characters (like ANSI color codes)
        with open(log_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        for line in lines:
            stripped_line = line.strip()

            if "Extracted Output String:" in stripped_line:
                found_json_start_marker = True
                # JSON block starts on the *next* line usually
                parsing_json = False # Reset in case of multiple markers
                json_string = ""
                brace_count = 0
                continue

            if found_json_start_marker and not parsing_json:
                # Look for the start of the JSON block
                if stripped_line.startswith('{'):
                    parsing_json = True
                    json_string += line # Start accumulating (keep original whitespace)
                    brace_count += line.count('{')
                    brace_count -= line.count('}')
                # Skip blank lines between marker and JSON start
                elif not stripped_line:
                    continue
                # If something else appears before '{', stop searching in this block
                else:
                    found_json_start_marker = False

            elif parsing_json:
                json_string += line
                brace_count += line.count('{')
                brace_count -= line.count('}')

                # Check if braces are balanced and we've likely found the end
                # This assumes the final '}' is on its own line or the last content line
                if brace_count <= 0 and stripped_line.endswith('}'):
                    try:
                        # Attempt to parse the accumulated string
                        parsed_output = json.loads(json_string)
                        identified_segments = parsed_output.get('identified_segments', [])
                        logging.info(f"Successfully parsed final JSON output and extracted {len(identified_segments)} segments.")
                        parsing_json = False # Done parsing this block
                        break # Found the JSON, no need to read further
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse accumulated JSON string: {e}\nString was:\n{json_string}")
                        # Reset and potentially look for another block if format is weird
                        parsing_json = False
                        json_string = ""
                        brace_count = 0
                        # Keep found_json_start_marker=True to potentially find another block later? Or just fail here? Let's fail here for now.
                        break

        # Log warnings if segments weren't found/parsed correctly
        if not identified_segments and found_json_start and not parsing_json and not json_string:
             logging.warning("Found 'Extracted Output String:' but no subsequent JSON block starting with '{' was found.")
        elif not identified_segments and found_json_start and parsing_json:
             logging.warning("Started parsing JSON block but did not find balanced braces or failed to parse.")
        elif not found_json_start_marker:
             logging.warning("Could not find 'Extracted Output String:' marker in log file.")

        return identified_segments
    except FileNotFoundError:
        logging.error(f"Log file not found: {log_filepath}")
        return []
    except Exception as e:
        logging.error(f"Error parsing log file {log_filepath}: {e}")
        return []


def get_segment_slice(full_df: pd.DataFrame, segment_dict: dict, column_name: str) -> np.ndarray:
    """Extracts the time-series data for a given segment dictionary and column."""
    try:
        start = segment_dict['start_row']
        end = segment_dict['end_row']
        if column_name not in full_df.columns:
            logging.warning(f"Column '{column_name}' not found in DataFrame. Cannot extract slice.")
            return None
        # Ensure indices are within bounds
        if not (0 <= start < len(full_df) and 0 <= end < len(full_df)):
             logging.warning(f"Segment indices [{start}-{end}] out of bounds for DataFrame length {len(full_df)}.")
             return None
        if start > end:
             logging.warning(f"Start row {start} > end row {end} for segment.")
             return None

        # Extract the specified column's data for the segment
        segment_data = full_df.loc[start:end, column_name].values # Use .loc for label-based indexing (assuming default integer index)
        return segment_data.reshape(-1, 1) # Reshape for tslearn compatibility (n_timestamps, n_features)
    except KeyError as e:
        logging.error(f"Error accessing segment {segment_dict}: Missing key {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error extracting slice for segment {segment_dict}: {e}")
        return None

def calculate_similarity(segments_data: list[np.ndarray]) -> tuple[float | None, float | None]:
    """Calculates average pairwise DTW and Euclidean distances between segments."""
    if len(segments_data) < 2:
        logging.info("Need at least two segments to calculate pairwise similarity.")
        return None, None

    dtw_distances = []
    euclidean_distances = []

    for i in range(len(segments_data)):
        for j in range(i + 1, len(segments_data)):
            seg1 = segments_data[i]
            seg2 = segments_data[j]

            # Ensure segments are not None and have compatible shapes for comparison
            if seg1 is None or seg2 is None:
                logging.warning(f"Skipping comparison involving None segment ({i} vs {j}).")
                continue

            # DTW requires non-empty arrays
            if seg1.size == 0 or seg2.size == 0:
                 logging.warning(f"Skipping comparison involving empty segment ({i} vs {j}).")
                 continue

            try:
                # Calculate DTW distance
                # Note: DTW might require segments of similar lengths or specific parameters.
                # Add error handling or length normalization if needed.
                dtw_dist = dtw(seg1, seg2)
                dtw_distances.append(dtw_dist)
            except Exception as e:
                logging.warning(f"Error calculating DTW between segment {i} and {j}: {e}")

            try:
                # Calculate Euclidean distance (requires segments of the same length)
                # Simple approach: Use only if lengths match, or consider padding/resampling
                if len(seg1) == len(seg2):
                    # Flatten arrays for euclidean distance if they are multi-dimensional (though should be 1D here)
                    euc_dist = euclidean(seg1.flatten(), seg2.flatten())
                    euclidean_distances.append(euc_dist)
                else:
                    logging.warning(f"Skipping Euclidean distance for segments {i} and {j} due to different lengths ({len(seg1)} vs {len(seg2)}).")
            except Exception as e:
                logging.warning(f"Error calculating Euclidean distance between segment {i} and {j}: {e}")


    avg_dtw = np.mean(dtw_distances) if dtw_distances else None
    avg_euclidean = np.mean(euclidean_distances) if euclidean_distances else None

    return avg_dtw, avg_euclidean

def main():
    parser = argparse.ArgumentParser(description="Calculate Similarity Metrics for Agent Output")
    parser.add_argument("log_filepath", help="Path to the agent's log file containing the output.")
    parser.add_argument("--dataset-path", required=True, help="Path to the full time-series dataset CSV file.")
    parser.add_argument("--column-name", required=True, help="Name of the column used for similarity analysis.")
    args = parser.parse_args()

    logging.info(f"Processing log file: {args.log_filepath}")
    logging.info(f"Using dataset: {args.dataset_path}")
    logging.info(f"Using column: {args.column_name}")

    # 1. Load full dataset
    full_df = load_full_dataset(args.dataset_path)
    if full_df is None:
        sys.exit(1) # Exit if dataset loading fails

    # 2. Parse identified segments from log
    identified_segments = parse_agent_output(args.log_filepath)
    if not identified_segments:
        logging.warning("No identified segments found in the log file. Cannot calculate similarity.")
        sys.exit(0) # Exit gracefully if no segments

    # 3. Extract data slices for each segment
    # Use the column name provided as an argument
    query_column = args.column_name
    logging.info(f"Extracting data slices for column: '{query_column}'")
    segment_slices = [get_segment_slice(full_df, seg, query_column) for seg in identified_segments]
    # Filter out None slices if errors occurred during extraction
    valid_segment_slices = [s for s in segment_slices if s is not None]

    if len(valid_segment_slices) < len(identified_segments):
         logging.warning(f"Could not extract data for {len(identified_segments) - len(valid_segment_slices)} segments.")

    if not valid_segment_slices:
         logging.error("No valid segment data could be extracted. Cannot calculate similarity.")
         sys.exit(1)

    # 4. Calculate pairwise similarity
    logging.info(f"Calculating pairwise similarity for {len(valid_segment_slices)} valid segments...")
    avg_dtw, avg_euclidean = calculate_similarity(valid_segment_slices)

    # 5. Output results
    print("\n--- Similarity Metrics (Metric A) ---")
    if avg_dtw is not None:
        print(f"Average Pairwise DTW Distance: {avg_dtw:.4f}")
    else:
        print("Average Pairwise DTW Distance: N/A (Could not be calculated)")

    if avg_euclidean is not None:
        print(f"Average Pairwise Euclidean Distance: {avg_euclidean:.4f}")
    else:
        print("Average Pairwise Euclidean Distance: N/A (Could not be calculated - check segment lengths)")
    print("------------------------------------")

if __name__ == "__main__":
    main()
