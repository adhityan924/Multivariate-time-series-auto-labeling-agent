import os
import sys
import json
from pprint import pprint

# Add src directory to Python path to allow importing modules from src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.api_functions import create_collection_from_file, find_similar_sequences
    # Import config to potentially use values if needed, though functions use it internally
    from src import config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you are running this script from the project root directory")
    print("and that the 'src' directory is correctly structured.")
    sys.exit(1)

# --- Test Configuration ---
# Use the same default dataset as the main CLI application for consistency
# Ideally, this could also be a config value or command-line argument
TEST_DATA_FILE_PATH = "dataset/#1-mini.csv"

# Sample sequence for testing the find_similar_sequences function
# Make it reasonably long, similar to potential user input
SAMPLE_SEQUENCE = [
    5.0, 5.1, 5.2, 4.9, 5.0, 5.3, 5.5, 5.4, 5.2, 5.1,
    5.0, 4.8, 4.7, 4.9, 5.0, 5.2, 5.4, 5.6, 5.5, 5.3,
    5.1, 5.0, 4.9, 4.8, 4.9, 5.1, 5.3, 5.5, 5.7, 5.6
]

# Optional: Define a specific top_k for testing, otherwise default from config will be used
TEST_TOP_K = 5

# --- Test Execution ---

def run_tests():
    """Runs tests for the API functions."""
    print("--- Starting API Function Tests ---")

    # Check if test data file exists
    if not os.path.exists(TEST_DATA_FILE_PATH):
        print(f"\nERROR: Test data file not found at '{TEST_DATA_FILE_PATH}'.")
        print("Please ensure the dataset exists in the 'dataset' folder.")
        print("--- Tests Aborted ---")
        return

    # 1. Test create_collection_from_file
    print(f"\n[TEST 1] Calling create_collection_from_file with: {TEST_DATA_FILE_PATH}")
    create_result = create_collection_from_file(TEST_DATA_FILE_PATH)
    print("\nResult from create_collection_from_file:")
    pprint(create_result)

    if not create_result.get("success"):
        print("\nERROR: create_collection_from_file failed. Aborting further tests.")
        print("--- Tests Failed ---")
        return

    print("\nCollection creation/update reported successful.")

    # 2. Test find_similar_sequences
    print(f"\n[TEST 2] Calling find_similar_sequences with sample sequence (length {len(SAMPLE_SEQUENCE)}) and top_k={TEST_TOP_K}")
    find_result = find_similar_sequences(sequence=SAMPLE_SEQUENCE, top_k=TEST_TOP_K)
    print("\nResult from find_similar_sequences:")
    # Pretty print, handling potential large 'results' dictionary
    if find_result.get("success"):
        print("  Success: True")
        # Optionally print only parts of the results if they are too large
        results_data = find_result.get("results", [])
        print(f"    Results found: {len(results_data)}")
        print(f"Results are: {results_data}")
        # print("    Full Results:")
        # pprint(results_data) # Uncomment to see full results
    else:
        pprint(find_result) # Print the full error dictionary

    if not find_result.get("success"):
        print("\nERROR: find_similar_sequences failed.")
        print("--- Tests Partially Failed ---")
        return

    # 3. Test find_similar_sequences with default top_k
    print(f"\n[TEST 3] Calling find_similar_sequences with sample sequence (length {len(SAMPLE_SEQUENCE)}) and default top_k (from config: {config.TOP_K})")
    find_result_default_k = find_similar_sequences(sequence=SAMPLE_SEQUENCE)
    print("\nResult from find_similar_sequences (default top_k):")
    if find_result_default_k.get("success"):
        print("  Success: True")
        results_data = find_result.get("results", [])
        print(f"    Results found: {len(results_data)}")
        print(f"Results are: {results_data}")
    else:
        pprint(find_result_default_k)

    if not find_result_default_k.get("success"):
        print("\nERROR: find_similar_sequences (default top_k) failed.")
        print("--- Tests Partially Failed ---")
        return


    print("\n--- All API Function Tests Completed Successfully ---")

if __name__ == "__main__":
    run_tests()
