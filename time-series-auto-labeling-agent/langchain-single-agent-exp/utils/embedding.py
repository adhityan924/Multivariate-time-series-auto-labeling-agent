import os
from typing import List, Callable
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

class ChromaEmbeddingFunction:
    def __init__(self, embedder):
        self.embedder = embedder

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embedder.embed_documents(input)

def get_embedding_model(model_name: str = "openai") -> Callable[[List[str]], List[List[float]]]:
    if model_name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
        embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
    elif model_name == "sentence-transformers":
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unknown embedding model: {model_name}")
    
    # Return an instance of our wrapper with the correct signature.
    return ChromaEmbeddingFunction(embedder)
