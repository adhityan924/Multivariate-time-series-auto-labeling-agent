"""
Utility functions for managing the vector database.
"""
import os
import json
from typing import List, Dict, Any, Optional, Callable

import chromadb
from chromadb.utils import embedding_functions
from chromadb.errors import InvalidCollectionException


from utils.display import print_info, print_success, print_error

class VectorDatabaseManager:
    """
    Manager for the vector database.
    """

    def __init__(
        self,
        collection_name: str = "time_series",
        persist_directory: str = "./chroma_db",
        embedding_function: Optional[Callable] = None,
    ):
        """
        Initialize the vector database manager.
        
        Args:
            collection_name: Name of the collection in the database.
            persist_directory: Directory to persist the database.
            embedding_function: Function to use for embedding documents.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        
        # Create the persistence directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize the client and collection
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create the collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            print_info(f"Using existing collection: {collection_name}")
        except InvalidCollectionException:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
        print_success(f"Created new collection: {collection_name}")


    def add_documents(
        self, ids: List[str], documents: List[str], metadatas: Optional[List[Any]] = None
    ) -> None:
        """
        Add documents to the vector database.
        
        Args:
            ids: List of document IDs.
            documents: List of document texts.
            metadatas: List of metadata for each document.
        """
        # Convert metadatas to strings if provided
        if metadatas is not None:
            metadatas = [{"chunk": json.dumps(metadata)} for metadata in metadatas]
        
        # Add documents to the collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def query(self, query_text: str, n_results: int = 3) -> List[List[float]]:
        """
        Query the vector database for similar documents.
        
        Args:
            query_text: Text to query.
            n_results: Number of results to return.
            
        Returns:
            List of similar time series chunks.
        """
        # Query the collection
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Extract metadatas
        chunks = []
        if results["metadatas"]:
            for metadata in results["metadatas"][0]:
                if "chunk" in metadata:
                    chunk = json.loads(metadata["chunk"])
                    chunks.append(chunk)
        
        return chunks 