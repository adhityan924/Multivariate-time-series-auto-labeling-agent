import pandas as pd
import numpy as np
from langchain_core.tools import tool
from colorama import Fore, Style
from typing import List

# Define patterns for columns to exclude (customize as needed)
EXCLUDED_PATTERNS = ['time', 'timestamp', 'index', 'id', 'record', 'unnamed']

@tool() # Added parentheses
def identify_timeseries_columns(df: pd.DataFrame) -> List[str]:
    """
    Identifies potential time series columns in a DataFrame.
    Selects numerical columns and filters out common non-timeseries names.

    Args:
        df: The input pandas DataFrame.

    Returns:
        A list of column names identified as potential time series data.
        Returns an empty list if no suitable columns are found or df is invalid.
    """
    print(f"{Fore.CYAN}Attempting to identify numerical time series columns...{Style.RESET_ALL}")
    if df is None or df.empty:
        print(f"{Fore.YELLOW}Warning: Input DataFrame is empty or None. Cannot identify columns.{Style.RESET_ALL}")
        return []

    # Select columns with numerical data types
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numerical_cols:
        print(f"{Fore.YELLOW}Warning: No numerical columns found in the DataFrame.{Style.RESET_ALL}")
        return []

    print(f"{Fore.BLUE}  Found numerical columns: {numerical_cols}{Style.RESET_ALL}")

    # Filter out columns based on excluded patterns
    columns_to_process = [
        col for col in numerical_cols
        if not any(pattern in col.lower() for pattern in EXCLUDED_PATTERNS)
    ]

    if not columns_to_process:
        print(f"{Fore.YELLOW}Warning: All numerical columns were filtered out based on exclusion patterns ({EXCLUDED_PATTERNS}).{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}Identified columns for processing: {columns_to_process}{Style.RESET_ALL}")

    return columns_to_process

# Note: Agent boilerplate (LLM, prompt, executor) is omitted here for simplicity,
# as the core logic is in the tool function, similar to the refined approach
# used in other agents in this project (calling the tool function directly).
# If complex reasoning were needed, the agent structure could be added.

# Function to be called from main.py (optional wrapper, could import tool directly)
def run_column_selection(df: pd.DataFrame) -> List[str]:
    """Runs the column identification logic using the invoke method."""
    # The @tool decorator turns identify_timeseries_columns into a Runnable.
    # We call it using .invoke() with a dictionary matching the argument name.
    try:
        # Assuming the tool expects a dictionary input matching its arguments
        return identify_timeseries_columns.invoke({"df": df})
    except Exception as e:
        print(f"{Fore.RED}Error invoking identify_timeseries_columns tool: {e}{Style.RESET_ALL}")
        # Depending on desired behavior, could raise the error or return empty list
        return []
