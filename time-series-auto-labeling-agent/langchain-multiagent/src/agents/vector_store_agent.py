import chromadb
import numpy as np
import os
from colorama import Fore, Style
from typing import List, Dict, Any

# NOTE: Removed Langchain agent boilerplate and @tool decorators as the logic
# is called directly from main.py via the functions below.

# --- Functions to be called from main script ---

def add_to_vector_store(
    collection: chromadb.Collection,
    embedding: List[float],
    chunk_data: np.ndarray, # Receive numpy array directly
    metadata: Dict[str, Any],
    doc_id: str
) -> bool:
    """
    Adds a single embedding, its corresponding chunk content (as string),
    and metadata to the specified ChromaDB collection.
    Uses a unique document ID. Returns True on success, False on failure.
    """
    if collection is None:
        print(f"{Fore.RED}ChromaDB collection not provided. Cannot add embedding.{Style.RESET_ALL}")
        return False
    if not embedding:
        print(f"{Fore.YELLOW}Skipping add to store: Empty embedding provided for ID {doc_id}.{Style.RESET_ALL}")
        return False

    print(f"{Fore.CYAN}Adding embedding to ChromaDB (ID: {doc_id}). Metadata: {metadata}{Style.RESET_ALL}")
    try:
        # Convert numpy array chunk to list, then to string for storage
        chunk_list = chunk_data.tolist()
        document_str = str(chunk_list)

        collection.add(
            embeddings=[embedding],
            documents=[document_str], # Store original chunk data as string
            metadatas=[metadata],
            ids=[doc_id]
        )
        # print(f"{Fore.GREEN}Successfully added embedding ID: {doc_id}{Style.RESET_ALL}") # Can be too verbose
        return True
    except Exception as e:
        print(f"{Fore.RED}Error adding embedding ID {doc_id} to ChromaDB: {e}{Style.RESET_ALL}")
        return False


def search_vector_store(
    collection: chromadb.Collection,
    query_embedding: List[float],
    top_k: int
) -> List[List[float]]:
    """
    Searches the ChromaDB collection for the top_k embeddings most similar
    to the query_embedding. Returns a list of the similar chunk data (as lists of floats).
    """
    if collection is None:
        print(f"{Fore.RED}ChromaDB collection not provided. Cannot search.{Style.RESET_ALL}")
        return []
    if not query_embedding:
        print(f"{Fore.YELLOW}Cannot search: Empty query embedding provided.{Style.RESET_ALL}")
        return []

    print(f"{Fore.CYAN}Searching for top {top_k} similar embeddings in ChromaDB.{Style.RESET_ALL}")
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents'] # We stored the original chunk as a string document
        )

        similar_chunks = []
        if results and results.get('documents') and results['documents'][0]:
            for doc_str in results['documents'][0]:
                try:
                    # Convert the string representation back to a list of floats
                    # This relies on the string being a valid list representation
                    # Using eval is generally unsafe, but necessary here if storing as str(list)
                    # Consider storing as JSON string in the future for safer parsing with json.loads
                    chunk_list = eval(doc_str)
                    if isinstance(chunk_list, list):
                        # Optional: Convert elements back to float if needed, though eval might handle it
                        # chunk_list = [float(x) for x in chunk_list]
                        similar_chunks.append(chunk_list)
                    else:
                         print(f"{Fore.YELLOW}Warning: Could not parse document string back to list: {doc_str[:50]}...{Style.RESET_ALL}")
                except Exception as parse_error:
                    # Catch errors during eval
                    print(f"{Fore.YELLOW}Warning: Could not parse document string '{doc_str[:50]}...': {parse_error}{Style.RESET_ALL}")

        print(f"{Fore.GREEN}Found {len(similar_chunks)} similar chunks.{Style.RESET_ALL}")
        return similar_chunks
    except Exception as e:
        print(f"{Fore.RED}Error searching ChromaDB: {e}{Style.RESET_ALL}")
        return []
