# Configuration settings for the time series similarity application

# Chunking parameters
CHUNK_SIZE = 50
OVERLAP = 10

# Search parameters
TOP_K = 3

# ChromaDB Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "timeseries_chunks"

# Embedding model details (if needed, e.g., for specific model selection)
# EMBEDDING_MODEL_NAME = "your_model_name_here" # Example if needed later
