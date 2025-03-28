import numpy as np
import chromadb

class DB:
    def __init__(self):
        # Initialize the ChromaDB client and collection
        self.client = chromadb.Client()
        # Create or get a collection named "time_series"
        try:
            self.collection = self.client.get_collection("time_series")
        except Exception:
            self.collection = self.client.create_collection("time_series")

    def add_embedding(self, embedding, chunk):
        """
        Add the embedding and corresponding chunk to the collection.
        Converts NumPy arrays to lists for JSON-serializability.
        """
        # Ensure both embedding and chunk are stored as lists
        embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding
        chunk_list = chunk.tolist() if hasattr(chunk, "tolist") else chunk

        # Generate an ID based on the current number of embeddings
        current_ids = self.collection.get()['ids']
        new_id = str(len(current_ids))

        # Store minimal metadata about the chunk
        metadata = {
            "chunk_length": len(chunk_list),
            "first_value": chunk_list[0] if len(chunk_list) > 0 else None,
            "last_value": chunk_list[-1] if len(chunk_list) > 0 else None
        }
        
        self.collection.add(
            embeddings=[embedding_list],
            metadatas=[metadata],
            ids=[new_id]
        )

    def query(self, query_embedding, top_k):
        """Query the collection for the top_k most similar embeddings."""
        query_embedding_list = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=top_k
        )
        return results

def get_db():
    return DB()
