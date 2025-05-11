# Configuration file for Experiment 2

import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# --- API Keys ---
# Load the OpenAI API key from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables. Some operations might fail.")

# --- File Paths ---
# Define relative paths from the project root (experiment2)
DATASET_PATH = "data/your_timeseries.csv"  # Placeholder - User needs to provide the actual dataset
KNOWLEDGE_PDF_PATH = "knowledge/paper.pdf"
LOG_DIR = "logs"
DB_DIR = "db/chroma_knowledge_db"

# --- Query Parameters ---
# Define the parameters for the specific query the agent should run
QUERY_START_ROW = 100  # Example value, adjust as needed
QUERY_END_ROW = 150    # Example value, adjust as needed
QUERY_COLUMN_NAME = "available_capacity (Ah)" # Example value, adjust based on dataset column names
QUERY_LABEL = "Pattern_A" # Example label to assign

# --- Logging Configuration ---
LOG_FILE_FORMAT = "experiment_2_run_{timestamp}.log"
LOG_LEVEL = "INFO" # e.g., DEBUG, INFO, WARNING, ERROR

print("config.py for Experiment 2 loaded.")
print(f"Dataset path set to: {DATASET_PATH}")
print(f"Knowledge PDF path set to: {KNOWLEDGE_PDF_PATH}")
print(f"Query parameters: Rows {QUERY_START_ROW}-{QUERY_END_ROW}, Column '{QUERY_COLUMN_NAME}', Label '{QUERY_LABEL}'") 