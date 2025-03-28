#!/usr/bin/env python
"""
CLI entry point for the Time Series Similarity and Annotation Agent.
"""
import os
import argparse
import json
from dotenv import load_dotenv

from agent import TimeSeriesAgent

def main():
    """
    Main function to run the Time Series Similarity and Annotation Agent CLI.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Time Series Similarity and Annotation Agent"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./dataset/#1-mini.csv",
        help="Path to the CSV file containing time series data",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Size of chunks for time series data",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=10,
        help="Overlap between chunks",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of similar time series chunks to return",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="openai",
        choices=["openai", "sentence-transformers"],
        help="Embedding model to use",
    )
    
    args = parser.parse_args()

    # Initialize the agent
    agent = TimeSeriesAgent(
        data_path=args.data_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
    )

    # Run the agent
    agent.run()

if __name__ == "__main__":
    main() 