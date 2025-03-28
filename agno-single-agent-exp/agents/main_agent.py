from agno.agent import Agent
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from colorama import Fore, Style
from utils.embedding_utils import embedding_utils  # Import the singleton instance
from utils import db_utils

class TimeSeriesAgent(Agent):
    def __init__(self):
        super().__init__()
        self.chunk_size = 50
        self.overlap = 10
        self.db = db_utils.get_db()
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load time series data from CSV file"""
        print(Fore.GREEN + "Loading data..." + Style.RESET_ALL)
        try:
            df = pd.read_csv(filepath)
            print(Fore.GREEN + "Data loaded successfully!" + Style.RESET_ALL)
            return df
        except Exception as e:
            print(Fore.RED + f"Error loading data: {e}" + Style.RESET_ALL)
            raise

    def chunk_time_series(self, series: np.ndarray) -> List[np.ndarray]:
        """Split time series into overlapping chunks"""
        print(Fore.BLUE + "Chunking time series..." + Style.RESET_ALL)
        chunks = []
        step = self.chunk_size - self.overlap
        for i in range(0, len(series) - self.chunk_size + 1, step):
            chunks.append(series[i:i + self.chunk_size])
        return chunks

    def process_dataset(self, df: pd.DataFrame) -> None:
        """Process entire dataset: chunk, embed and store in DB"""
        print(Fore.CYAN + "Processing dataset..." + Style.RESET_ALL)
        for column in df.columns:
            if df[column].dtype in [np.float64, np.int64]:
                chunks = self.chunk_time_series(df[column].values)
                for chunk in chunks:
                    embedding = embedding_utils.get_embedding(chunk)
                    self.db.add_embedding(embedding, chunk)
    
    def query(self, query_series: List[float], label: str, top_k: int = 3) -> Dict[str, Any]:
        """Query similar time series patterns"""
        # Only embed the numerical time series data
        print(query_series)
        query_embedding = embedding_utils.get_embedding(np.array(query_series))
        
        # Query using just the numerical embedding
        results = self.db.query(query_embedding, top_k)
        print("DEBUG - Raw ChromaDB results:", results)  # Add debug print
        
        print("DEBUG - Full results structure:", results)
        print("DEBUG - Metadatas content:", results['metadatas'])
        print("DEBUG - First metadata item:", results['metadatas'][0][0])
        
        # Use label purely for organizing results
        formatted_results = {label: [(m['first_value'], m['last_value']) 
                              for m in results['metadatas'][0]]}
        print("DEBUG - Formatted results:", formatted_results)
        return formatted_results
