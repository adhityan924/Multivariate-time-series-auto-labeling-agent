from agents.main_agent import TimeSeriesAgent
from colorama import Fore, Style, init
import argparse
import json
import os

def main():
    init()  # Initialize colorama
    agent = TimeSeriesAgent()
    
    # Load and process dataset
    data_path = "dataset/#1-mini.csv"
    df = agent.load_data(data_path)
    agent.process_dataset(df)
    
    # Get user input
    print(Fore.MAGENTA + "\nEnter your time series pattern to search for:" + Style.RESET_ALL)
    while True:
        try:
            ts_input = input("Time series array (comma-separated numbers): ")
            ts_array = [float(x.strip()) for x in ts_input.split(",")]
            
            label = input("Annotation label: ").strip()
            
            # Query similar patterns
            results = agent.query(ts_array, label)
            
            # Print results
            print(Fore.CYAN + "\nSimilar patterns found:" + Style.RESET_ALL)
            print(json.dumps(results, indent=2))
            
            # Ask to continue
            cont = input("\nSearch again? (y/n): ").lower()
            if cont != 'y':
                break
                
        except Exception as e:
            print(Fore.RED + f"Error: {e}" + Style.RESET_ALL)
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Similarity Search")
    parser.add_argument("--local", action="store_true", help="Use local embeddings instead of OpenAI")
    args = parser.parse_args()
    
    if args.local:
        from utils.embedding_utils import EmbeddingUtils
        EmbeddingUtils(use_openai=False)
    
    main()
