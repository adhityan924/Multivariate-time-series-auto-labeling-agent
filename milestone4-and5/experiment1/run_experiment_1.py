# Main execution script for Experiment 1

import logging
import os
import datetime
import json
import sys
import os

# Add the project root directory (parent of 'langchain-experiment1') to the Python path
# This allows importing from the top-level 'src' and 'evaluation' directories
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import project modules
import config # Still relative to this script's directory
from src.tools import load_data, setup_vector_store # Now imports from top-level src
from src.agent_setup import agent_executor # Direct import from local src directory
from langchain.callbacks.tracers.logging import LoggingCallbackHandler # Ensure import is present

# --- Logging Setup ---
def setup_logging():
    """Configures logging to console and a timestamped file."""
    log_dir = config.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = config.LOG_FILE_FORMAT.format(timestamp=timestamp)
    log_filepath = os.path.join(log_dir, log_filename)

    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers (if any)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    logging.info(f"Logging setup complete. Log file: {log_filepath}")
    return log_filepath # Return the path for potential use later

# --- Main Execution Logic ---
def run_experiment():
    """Loads data, initializes components, runs the agent with logging callback, and logs results."""
    log_filepath = setup_logging()
    logger = logging.getLogger() # Get the root logger configured in setup_logging
    # Ensure the handler uses the same logger and an appropriate level (e.g., INFO or DEBUG)
    log_handler = LoggingCallbackHandler(logger=logger, log_level=logging.INFO)
    logging.info("Starting Experiment 1 Run...")

    # --- Pre-computation/Setup ---
    # 1. Load Dataset (using the tool function directly for setup)
    logging.info(f"Attempting to load dataset from: {config.DATASET_PATH}")
    load_result = load_data(config.DATASET_PATH)
    if "Error" in load_result:
        logging.error(f"Failed to load dataset: {load_result}. Exiting.")
        return
    logging.info("Dataset loaded successfully.")

    # 2. Initialize Vector Store (triggers embedding if needed)
    logging.info("Initializing vector store...")
    retriever = setup_vector_store()
    if not retriever:
        logging.error("Failed to initialize vector store. Exiting.")
        return
    logging.info("Vector store initialized successfully.")

    # 3. Check if Agent Executor is ready
    if not agent_executor:
        logging.error("Agent Executor not initialized correctly in agent_setup.py. Exiting.")
        return
    logging.info("Agent Executor is ready.")

    # --- Agent Invocation ---
    # 4. Construct Input for the Agent
    # The ReAct prompt expects specific input variables. We format them here.
    # The main query goes into the 'input' key.
    # Other parameters from config are used to fill the prompt template placeholders.
    input_query = (
        f"Find segments similar to the one from row {config.QUERY_START_ROW} to {config.QUERY_END_ROW} "
        f"in column '{config.QUERY_COLUMN_NAME}', and label them as '{config.QUERY_LABEL}'."
    )

    agent_input = {
        "input": input_query,
        "input_start_row": config.QUERY_START_ROW,
        "input_end_row": config.QUERY_END_ROW,
        "input_column_name": config.QUERY_COLUMN_NAME,
        "input_label": config.QUERY_LABEL,
        # agent_scratchpad is handled internally by AgentExecutor
    }

    logging.info(f"Invoking agent with input: {agent_input}")

    try:
        # 5. Run the Agent with the callback handler passed in the config
        response = agent_executor.invoke(agent_input, config={"callbacks": [log_handler]})
        logging.info("Agent execution completed.")
        logging.info(f"Raw Agent Response: {response}")

        # --- Output Parsing & Logging (Task 9) ---
        # Placeholder for Task 9 implementation
        logging.info("--- Agent Final Output ---")
        if isinstance(response, dict) and 'output' in response:
             final_output_str = response['output']
             logging.info(f"Extracted Output String:\n{final_output_str}")
             # Attempt to parse the final output string as JSON
             try:
                 # Clean the string if necessary (e.g., remove potential markdown backticks)
                 if final_output_str.startswith("```json"):
                     final_output_str = final_output_str[7:]
                 if final_output_str.endswith("```"):
                     final_output_str = final_output_str[:-3]
                 final_output_str = final_output_str.strip()

                 parsed_output = json.loads(final_output_str)
                 logging.info("Successfully parsed final output JSON.")

                 # Log the structured components
                 segments = parsed_output.get('identified_segments', [])
                 label = parsed_output.get('assigned_label', 'N/A')
                 explanation = parsed_output.get('explanation', 'N/A')
                 uncertainty = parsed_output.get('uncertainty_notes', 'N/A')

                 logging.info(f"Identified Segments ({len(segments)}):")
                 if segments:
                     for segment in segments:
                         logging.info(f"  - Start: {segment.get('start_row', '?')}, End: {segment.get('end_row', '?')}")
                 else:
                     logging.info("  (None)")

                 logging.info(f"Assigned Label: {label}")
                 logging.info(f"Explanation: {explanation}")
                 logging.info(f"Uncertainty Notes: {uncertainty}")

             except json.JSONDecodeError as json_error:
                 logging.error(f"Failed to parse final output string as JSON: {json_error}")
                 logging.warning("Agent did not return the expected JSON format in 'output'. The raw string was logged above.")
             except Exception as parse_exc:
                 logging.error(f"An unexpected error occurred during output parsing: {parse_exc}", exc_info=True)
        else:
             logging.warning("Agent response format unexpected or missing 'output' key.")
             logging.info(f"Full Response: {response}")


    except Exception as e:
        logging.error(f"An error occurred during agent execution: {e}", exc_info=True)

    logging.info("Experiment 1 Run Finished.")
    return log_filepath # Return log path for evaluation scripts

if __name__ == "__main__":
    run_experiment()
