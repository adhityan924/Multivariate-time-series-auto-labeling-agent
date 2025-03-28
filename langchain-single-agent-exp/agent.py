"""
Time Series Similarity and Annotation Agent using the langchain framework.
"""
import os
import json
import ast
from typing import List, Dict, Any, Optional

from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.tools import Tool
import json

from utils.chunking import chunk_time_series
from utils.embedding import get_embedding_model
from utils.database import VectorDatabaseManager
from utils.display import (
    print_info, print_success, print_error, print_warning, print_step
)

class TimeSeriesAgent:
    """
    Agent for time series similarity and annotation tasks.
    """

    def __init__(
        self,
        data_path: str,
        chunk_size: int = 50,
        overlap: int = 10,
        top_k: int = 3,
        embedding_model: str = "openai",
    ):
        """
        Initialize the time series agent.

        Args:
            data_path: Path to the CSV file containing time series data.
            chunk_size: Size of chunks for time series data.
            overlap: Overlap between chunks.
            top_k: Number of similar time series chunks to return.
            embedding_model: Embedding model to use ("openai" or "sentence-transformers").
        """
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.embedding_model = embedding_model
        
        # Initialize components
        self.db_manager = None
        self.embedder = None
        
        # Initialize the agent components
        self._initialize_agent()

    def _initialize_agent(self):
        """
        Initialize the agent components.
        """
        # Initialize embedding model
        self.embedder = get_embedding_model(self.embedding_model)
        
        # Initialize vector database manager
        self.db_manager = VectorDatabaseManager(embedding_function=self.embedder)

    def load_data(self) -> str:
        """
        Load time series data from the specified path.
        """
        import pandas as pd
        
        print_step("Loading data from:", self.data_path)
        
        try:
            data = pd.read_csv(self.data_path)
            print_success(f"Data loaded successfully with {len(data)} rows and {len(data.columns)} columns.")
            print_info(f"Columns: {', '.join(data.columns)}")
            return json.dumps({
                "status": "success",
                "message": f"Data loaded successfully with {len(data)} rows and {len(data.columns)} columns.",
                "columns": list(data.columns),
                "shape": data.shape
            })
        except Exception as e:
            print_error(f"Error loading data: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": f"Error loading data: {str(e)}"
            })

    def process_data(self) -> str:
        """
        Process time series data by chunking each column and embedding the chunks.
        """
        import pandas as pd
        import numpy as np
        
        print_step("Processing data")
        
        try:
            # Load data
            data = pd.read_csv(self.data_path)
            
            # Get all columns except 'record_time' (if it exists)
            time_series_columns = [col for col in data.columns if col != 'record_time']
            
            # Chunk and embed each column
            all_chunks = {}
            for column in time_series_columns:
                print_info(f"Chunking column: {column}")
                # Convert column data to list of floats
                column_data = data[column].fillna(0).tolist()
                # Chunk the column data
                chunks = chunk_time_series(column_data, self.chunk_size, self.overlap)
                all_chunks[column] = chunks
                
                # Process the chunks in batches to avoid memory issues
                batch_size = 100
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    # Create unique IDs for each chunk
                    chunk_ids = [f"{column}_{i+j}" for j in range(len(batch))]
                    # Convert chunks to strings for embedding and storage
                    chunk_strs = [json.dumps(chunk) for chunk in batch]
                    
                    # Add to vector database
                    self.db_manager.add_documents(chunk_ids, chunk_strs, batch)
                
            # Get total number of chunks
            total_chunks = sum(len(chunks) for chunks in all_chunks.values())
            
            print_success(f"Data processing complete. Created {total_chunks} chunks across {len(time_series_columns)} columns.")
            
            return json.dumps({
                "status": "success",
                "message": f"Data processing complete. Created {total_chunks} chunks across {len(time_series_columns)} columns.",
                "chunks_per_column": {col: len(chunks) for col, chunks in all_chunks.items()}
            })
            
        except Exception as e:
            print_error(f"Error processing data: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": f"Error processing data: {str(e)}"
            })

    def query_database(self, query_data: Dict[str, Any]) -> List[List[float]]:
        """
        Query the database for similar time series chunks.
        
        Args:
            query_data: Dictionary containing time series array and annotation label.
            
        Returns:
            List of similar time series chunks.
        """
        time_series_array = query_data.get("time_series_array", [])
        
        # Convert the query array to string format
        query_str = json.dumps(time_series_array)
        
        # Query the database
        results = self.db_manager.query(query_str, n_results=self.top_k)
        
        return results

    def get_user_input(self) -> str:
        """
        Get input from the user (time series array and annotation label).
        """
        print_step("Getting user input")
        
        try:
            # Get time series array from user
            time_series_input = input("Enter time series array (e.g., [5, 5.1, 5.3, 5.2]): ")
            
            # Parse the input string to a list
            try:
                time_series_array = ast.literal_eval(time_series_input)
                if not isinstance(time_series_array, list):
                    raise ValueError("Input must be a list")
            except (ValueError, SyntaxError) as e:
                print_error(f"Invalid time series array: {str(e)}")
                return json.dumps({
                    "status": "error",
                    "message": f"Invalid time series array: {str(e)}"
                })
            
            # Get annotation label from user
            annotation_label = input("Enter annotation label (e.g., peak-pattern): ")
            
            print_success("User input received successfully.")
            
            return json.dumps({
                "status": "success",
                "time_series_array": time_series_array,
                "annotation_label": annotation_label
            })
            
        except Exception as e:
            print_error(f"Error getting user input: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": f"Error getting user input: {str(e)}"
            })

    def format_results(self, results: List[List[float]], annotation_label: str) -> str:
        """
        Format and return the query results.
        
        Args:
            results: List of time series chunks.
            annotation_label: Label to annotate the results with.
            
        Returns:
            JSON string containing formatted results.
        """
        print_step("Formatting results")
        
        try:
            output = {annotation_label: results}
            output_json = json.dumps(output, indent=2)
            print_success("Results formatted successfully.")
            return output_json
        except Exception as e:
            print_error(f"Error formatting results: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": f"Error formatting results: {str(e)}"
            })

    def run(self):
        """
        Run the agent to process time series data, get user input, and return similar chunks.
        """
        try:
            # Load the data
            print_step("Loading and processing data")
            load_result = json.loads(self.load_data())
            if load_result["status"] != "success":
                print_error("Failed to load data. Please check the data path.")
                return
                
            # Process the data
            process_result = json.loads(self.process_data())
            if process_result["status"] != "success":
                print_error("Failed to process data.")
                return
            
            # Get user input
            user_input_result = json.loads(self.get_user_input())
            if user_input_result["status"] != "success":
                print_error("Failed to get user input.")
                return
            
            # Query the database using the user's time series array
            query_result = json.loads(self.query_database(user_input_result))
            if query_result["status"] != "success":
                print_error("Failed to query the database.")
                return
            
            # Format the results
            result = self.format_results(query_result["results"], user_input_result["annotation_label"])
            
            # Print the final output
            print("\n" + "="*50)
            print(result)
            print("="*50 + "\n")
            
        except Exception as e:
            print_error(f"An error occurred: {str(e)}")
            import traceback
            print_error(traceback.format_exc())

if __name__ == "__main__":
    # Update the path to your data CSV file as needed.
    DATA_PATH = "./dataset/#1-mini.csv"
    agent = TimeSeriesAgent(data_path=DATA_PATH)
    agent.run()
