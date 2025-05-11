import os
import sys
import chromadb
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

# Ensure src directory is in path (useful if run standalone, though typically imported)
# sys.path.append(os.path.dirname(os.path.dirname(__file__))) # Add parent dir ('src') to path

try:
    from . import config  # Relative import if used as part of the package
    from .agents.data_loader_agent import load_csv_data
    from .agents.column_selection_agent import run_column_selection
    from .utils.chunking_utils import create_overlapping_chunks
    from .utils.embedding_utils import compute_embedding
    from .agents.vector_store_agent import add_to_vector_store, search_vector_store
except ImportError:
    # Fallback for potential direct execution or different environment setup
    print("Attempting absolute imports due to relative import failure...")
    import config
    from agents.data_loader_agent import load_csv_data
    from agents.column_selection_agent import run_column_selection
    from utils.chunking_utils import create_overlapping_chunks
    from utils.embedding_utils import compute_embedding
    from agents.vector_store_agent import add_to_vector_store, search_vector_store


def create_collection_from_file(file_path: str) -> Dict[str, Any]:
    """
    Loads data from a CSV file, processes it (chunks, embeds), and stores it
    in a new ChromaDB collection, replacing any existing collection with the same name.

    Args:
        file_path: The path to the input CSV file.

    Returns:
        A dictionary indicating success or failure, along with details like
        processed columns and the number of embeddings added.
    """
    print(f"--- API Function: create_collection_from_file ---")
    print(f"Input file path: {file_path}")

    client = None
    try:
        # 1. Initialize ChromaDB Client
        print(f"Initializing ChromaDB client at: {config.CHROMA_DB_PATH}")
        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

        # 2. Ensure collection is replaced: Try deleting it first, ignore error if it doesn't exist.
        try:
            print(f"Attempting to delete existing collection (if any): {config.COLLECTION_NAME}")
            client.delete_collection(name=config.COLLECTION_NAME)
            print(f"Collection '{config.COLLECTION_NAME}' deleted.")
        except Exception as e:
            # Catching a broad exception, but specifically expecting errors if collection doesn't exist.
            # ChromaDB might raise specific errors like ValueError or similar if not found,
            # but catching Exception is safer for now unless specific error types are known.
            print(f"Note: Collection '{config.COLLECTION_NAME}' likely did not exist or deletion failed: {e}. Proceeding to create.")

        # 3. Create the collection
        print(f"Creating collection: {config.COLLECTION_NAME}")
        collection = client.create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"} # Specify cosine distance
        )
        print(f"Collection '{config.COLLECTION_NAME}' created successfully.")

        # 4. Load Data
        print(f"Loading data from: {file_path}")
        df = load_csv_data(file_path)
        if df.empty:
            return {"success": False, "error": f"Failed to load data or data is empty from {file_path}"}
        print(f"Data loaded successfully. Shape: {df.shape}")

        # 5. Select Columns
        print("Running column selection...")
        columns_to_process = run_column_selection(df)
        if not columns_to_process:
            return {"success": False, "error": "No suitable columns identified for processing."}
        print(f"Columns selected for processing: {columns_to_process}")

        # 6. Process Each Column: Chunk -> Embed -> Store
        total_embeddings_added = 0
        processed_columns_list = []
        print("Starting data processing (Chunk -> Embed -> Store)...")
        for col_name in columns_to_process:
            print(f"--> Processing column: {col_name}")
            if col_name not in df.columns:
                 print(f"Warning: Column '{col_name}' missing. Skipping.")
                 continue

            column_data = df[col_name].dropna().to_numpy()
            if column_data.size < config.CHUNK_SIZE:
                print(f"Skipping column '{col_name}': Not enough data points ({column_data.size}) for chunk size ({config.CHUNK_SIZE}).")
                continue

            # Chunking
            print(f"  Chunking column '{col_name}'...")
            chunks = list(create_overlapping_chunks(column_data, config.CHUNK_SIZE, config.OVERLAP))
            if not chunks:
                print(f"  No chunks generated for column '{col_name}'.")
                continue
            print(f"  Generated {len(chunks)} chunks.")

            # Embedding & Storing
            print(f"  Embedding and storing {len(chunks)} chunks...")
            embeddings_added_for_col = 0
            for i, chunk in enumerate(chunks):
                embedding = compute_embedding(chunk)
                if embedding:
                    doc_id = f"{col_name}_chunk_{i}"
                    metadata = {
                        "column": col_name,
                        "chunk_index": i,
                        "start_row": i * (config.CHUNK_SIZE - config.OVERLAP),
                        "end_row": i * (config.CHUNK_SIZE - config.OVERLAP) + config.CHUNK_SIZE
                    }
                    success = add_to_vector_store(collection, embedding, chunk, metadata, doc_id)
                    if success:
                        embeddings_added_for_col += 1
                else:
                    print(f"  Skipping storage for chunk {i} in column '{col_name}' due to embedding failure.")

            print(f"  Finished processing column '{col_name}'. Embeddings added: {embeddings_added_for_col}")
            if embeddings_added_for_col > 0:
                processed_columns_list.append(col_name)
                total_embeddings_added += embeddings_added_for_col

        print("--- Data processing complete ---")
        if total_embeddings_added == 0:
             print("Warning: No embeddings were successfully added.")

        return {
            "success": True,
            "message": f"Collection '{config.COLLECTION_NAME}' created/updated successfully.",
            "columns_processed": processed_columns_list,
            "embeddings_added": total_embeddings_added
        }

    except FileNotFoundError:
        return {"success": False, "error": f"Input file not found: {file_path}"}
    except pd.errors.EmptyDataError:
        return {"success": False, "error": f"Input file is empty: {file_path}"}
    except Exception as e:
        print(f"An unexpected error occurred in create_collection_from_file: {e}")
        # Consider logging the full traceback here
        return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}


def find_similar_sequences(sequence: List[float], top_k: Optional[int] = None) -> Dict[str, Any]:
    """
    Finds time series sequences in the ChromaDB collection that are similar
    to the provided input sequence.

    Args:
        sequence: A list of floats representing the time series sequence.
        top_k: The number of similar sequences to return. If None, uses
               the default value from config.

    Returns:
        A dictionary containing the search results or an error message.
    """
    print(f"--- API Function: find_similar_sequences ---")
    print(f"Input sequence length: {len(sequence)}")
    print(f"Requested top_k: {top_k}")

    client = None
    try:
        # 1. Initialize ChromaDB Client and Get Collection
        print(f"Initializing ChromaDB client at: {config.CHROMA_DB_PATH}")
        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        print(f"Getting collection: {config.COLLECTION_NAME}")
        collection = client.get_collection(name=config.COLLECTION_NAME)
        print(f"Collection '{config.COLLECTION_NAME}' retrieved.")

        # 2. Process Input Sequence (Pad/Truncate)
        query_array = np.array(sequence, dtype=float)
        if query_array.size == 0:
            return {"success": False, "error": "Input sequence cannot be empty."}

        padded_query_array = query_array
        if len(query_array) < config.CHUNK_SIZE:
            print(f"Padding query array (length {len(query_array)}) to match chunk size ({config.CHUNK_SIZE}).")
            padding_size = config.CHUNK_SIZE - len(query_array)
            padded_query_array = np.pad(query_array, (0, padding_size), mode='constant')
            print(f"Padded query array length: {len(padded_query_array)}")
        elif len(query_array) > config.CHUNK_SIZE:
             print(f"Warning: Query length ({len(query_array)}) > chunk size ({config.CHUNK_SIZE}). Truncating.")
             padded_query_array = query_array[:config.CHUNK_SIZE]

        # 3. Embed Query
        print("Embedding query sequence...")
        query_embedding = compute_embedding(padded_query_array)
        if not query_embedding:
            return {"success": False, "error": "Failed to generate embedding for the query sequence."}
        print("Query embedding generated.")

        # 4. Determine K and Search
        k = top_k if top_k is not None and top_k > 0 else config.TOP_K
        print(f"Searching for top {k} similar sequences...")
        # search_vector_store returns a List[List[float]]
        similar_chunks_list = search_vector_store(collection, query_embedding, k)
        print(f"Search complete. Found {len(similar_chunks_list)} results.")

        # 5. Format Results (Return the list of similar chunks)
        return {"success": True, "results": similar_chunks_list}

    except ValueError as e: # Catch potential numpy errors or issues during client/collection access
        # Check if the error message indicates a missing collection
        if "does not exist" in str(e).lower() or "not found" in str(e).lower():
             return {"success": False, "error": f"Collection '{config.COLLECTION_NAME}' not found or error accessing it: {str(e)}"}
        else:
            return {"success": False, "error": f"Invalid input or processing error: {str(e)}"}
    except Exception as e: # Catch other unexpected errors
        print(f"An unexpected error occurred in find_similar_sequences: {e}")
        # Check if it's related to collection access
        if "does not exist" in str(e).lower() or "not found" in str(e).lower():
             return {"success": False, "error": f"Collection '{config.COLLECTION_NAME}' not found or error accessing it: {str(e)}"}
        # Consider logging the full traceback here for debugging
        return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}

# Example Usage (Optional - for testing)
if __name__ == '__main__':
    print("Running example usage of API functions...")

    # Ensure dataset path is correct relative to this file's location if run directly
    # This assumes the script is run from the project root directory
    example_file = os.path.join("dataset", "#1-mini.csv")
    if not os.path.exists(example_file):
         print(f"Error: Example dataset file not found at {example_file}. Make sure you run this from the project root.")
    else:
        # Example: Create collection
        print("\n--- Testing create_collection_from_file ---")
        create_result = create_collection_from_file(example_file)
        print("Create Result:", create_result)

        # Example: Find similar sequences (only if creation was successful)
        if create_result.get("success"):
            print("\n--- Testing find_similar_sequences ---")
            # Use a sample sequence (adjust length as needed)
            example_sequence = [5.0, 5.1, 5.2, 4.9, 5.0, 5.3] * 5 # Example sequence
            find_result = find_similar_sequences(example_sequence, top_k=2)
            print("Find Result:", find_result)

            # Example: Test with empty sequence
            print("\n--- Testing find_similar_sequences (empty) ---")
            find_empty_result = find_similar_sequences([])
            print("Find Empty Result:", find_empty_result)

            # Example: Test with non-existent collection (delete it first)
            try:
                print("\n--- Testing find_similar_sequences (collection deleted) ---")
                client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
                client.delete_collection(name=config.COLLECTION_NAME)
                print(f"Collection {repr(config.COLLECTION_NAME)} deleted for testing.")
                find_deleted_result = find_similar_sequences(example_sequence)
                print("Find Deleted Result:", find_deleted_result)
                # Recreate for subsequent tests if needed
                # create_collection_from_file(example_file)
            except Exception as e:
                print(f"Error during deletion test: {e}")
