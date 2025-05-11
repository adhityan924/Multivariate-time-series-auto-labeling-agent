# Main execution script for Experiment 2: Plan-and-Solve Agent

import logging
import os
import datetime
import json
import sys
import traceback

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import project modules
import config  # Import from the root directory of experiment2
from src.tools import load_data, setup_vector_store
from experiment2.src.agent_setup import create_plan_and_solve_agent, create_agent_executor, ps_prompt_template
from langchain.callbacks.tracers.logging import LoggingCallbackHandler

# Import json5 for more resilient JSON parsing
try:
    import json5
    HAS_JSON5 = True
except ImportError:
    HAS_JSON5 = False
    logging.warning("json5 module not found. Using standard json module only.")

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
    """Loads data, initializes components, runs the Plan-and-Solve agent with logging callback, and logs results."""
    log_filepath = setup_logging()
    logger = logging.getLogger() # Get the root logger configured in setup_logging
    # Ensure the handler uses the same logger and an appropriate level (e.g., INFO or DEBUG)
    log_handler = LoggingCallbackHandler(logger=logger, log_level=logging.INFO)
    logging.info("Starting Experiment 2 Run (Plan-and-Solve Agent)...")
    
    # Get paths for experiment2 resources
    experiment2_path = os.path.abspath(os.path.dirname(__file__))
    pdf_path = os.path.join(experiment2_path, "knowledge", "paper.pdf")
    db_path = os.path.join(experiment2_path, "db", "chroma_knowledge_db")
    
    # Log paths
    logging.info(f"Using experiment2 paths:")
    logging.info(f"- PDF Path: {pdf_path}")
    logging.info(f"- DB Path: {db_path}")

    # --- Pre-computation/Setup ---
    # 1. Load Dataset (using the tool function directly for setup)
    logging.info(f"Attempting to load dataset from: {config.DATASET_PATH}")
    load_result = load_data(config.DATASET_PATH)
    if "Error" in load_result:
        logging.error(f"Failed to load dataset: {load_result}. Exiting.")
        return
    logging.info("Dataset loaded successfully.")

    # 2. Initialize Vector Store with custom paths (triggers embedding if needed)
    logging.info("Initializing vector store...")
    retriever = setup_vector_store(knowledge_pdf_path=pdf_path, db_persist_path=db_path)
    if not retriever:
        logging.error("Failed to initialize vector store. Exiting.")
        return
    logging.info("Vector store initialized successfully.")
    
    # 3. Create the Plan-and-Solve agent
    logging.info("Creating Plan-and-Solve agent...")
    agent = create_plan_and_solve_agent(ps_prompt_template)
    if not agent:
        logging.error("Failed to create Plan-and-Solve agent. Exiting.")
        return
    logging.info("Plan-and-Solve agent created successfully.")
    
    # 4. Create the AgentExecutor
    logging.info("Creating AgentExecutor...")
    agent_executor = create_agent_executor(agent)
    if not agent_executor:
        logging.error("Failed to create AgentExecutor. Exiting.")
        return
    logging.info("AgentExecutor created successfully.")

    # --- Agent Invocation ---
    # 5. Construct Input for the Agent
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
    
    # Add special markers for Plan-and-Solve phases
    logging.info("=== PLANNING PHASE STARTING ===")

    try:
        # 6. Run the Agent with the callback handler passed in the config
        response = agent_executor.invoke(agent_input, config={"callbacks": [log_handler]})
        
        # Add marker for the transition to execution phase
        # Note: This is a simplification since the tool interface doesn't allow real-time monitoring
        # In a real implementation, we would inject markers based on agent's actual phase transitions
        logging.info("=== EXECUTION PHASE STARTING ===")
        
        logging.info("Agent execution completed.")
        logging.info(f"Raw Agent Response: {response}")

        # --- Output Parsing & Logging ---
        logging.info("--- Agent Final Output ---")
        if isinstance(response, dict) and 'output' in response:
             final_output_str = response['output']
             logging.info(f"Extracted Raw Output String:\n{final_output_str}")
             
             # Extract planning and execution phases
             try:
                 # Look for planning phase indicators
                 planning_indicators = ["PLANNING PHASE", "Initial Problem Analysis", "Task Decomposition", 
                                       "Strategy Formulation", "Execution Plan", "Reflection on Plan"]
                 execution_indicators = ["EXECUTION PHASE", "executing plan", "following my plan"]
                 
                 # Extract planning phase content
                 planning_content = ""
                 execution_content = ""
                 final_answer_content = ""
                 
                 # First, check if there's a "Final Answer:" marker
                 final_answer_index = final_output_str.find("Final Answer:")
                 
                 if final_answer_index != -1:
                     # Everything before Final Answer could be planning and execution
                     pre_final = final_output_str[:final_answer_index].strip()
                     final_answer_content = final_output_str[final_answer_index:].strip()
                     
                     # Try to find the boundary between planning and execution
                     execution_start_index = -1
                     for indicator in execution_indicators:
                         idx = pre_final.lower().find(indicator.lower())
                         if idx != -1 and (execution_start_index == -1 or idx < execution_start_index):
                             execution_start_index = idx
                     
                     if execution_start_index != -1:
                         planning_content = pre_final[:execution_start_index].strip()
                         execution_content = pre_final[execution_start_index:].strip()
                     else:
                         # Couldn't find a clear execution boundary, do our best guess
                         for i, line in enumerate(pre_final.split('\n')):
                             line_lower = line.lower()
                             if any(indicator.lower() in line_lower for indicator in planning_indicators):
                                 planning_content += line + '\n'
                             elif any(indicator.lower() in line_lower for indicator in execution_indicators):
                                 execution_content += line + '\n'
                             elif planning_content and not execution_content:
                                 # Default to planning until we see execution markers
                                 planning_content += line + '\n'
                             else:
                                 execution_content += line + '\n'
                 else:
                     # If no Final Answer marker, do basic split on content
                     lines = final_output_str.split('\n')
                     in_planning = True
                     
                     for line in lines:
                         line_lower = line.lower()
                         
                         # Check if we're transitioning to execution phase
                         if in_planning and any(indicator.lower() in line_lower for indicator in execution_indicators):
                             in_planning = False
                         
                         # Add to the appropriate phase
                         if in_planning:
                             planning_content += line + '\n'
                         else:
                             execution_content += line + '\n'
                 
                 # Log the extracted phases
                 if planning_content:
                     logging.info("=== EXTRACTED PLANNING PHASE ===")
                     logging.info(planning_content)
                 
                 if execution_content:
                     logging.info("=== EXTRACTED EXECUTION PHASE ===")
                     logging.info(execution_content)
                 
                 if final_answer_content:
                     logging.info("=== EXTRACTED FINAL ANSWER ===")
                     logging.info(final_answer_content)
             
             except Exception as phase_extract_error:
                 logging.error(f"Error extracting planning/execution phases: {phase_extract_error}")
                 logging.info("Continuing with JSON extraction anyway...")
             
             # Extract and parse the JSON output
             try:
                 # Extract just the JSON part after "Final Answer:"
                 json_start = final_output_str.find("Final Answer:")
                 if json_start != -1:
                     # Extract everything after "Final Answer:"
                     json_text = final_output_str[json_start + len("Final Answer:"):].strip()
                     
                     # Clean the string if necessary (e.g., remove potential markdown backticks)
                     if json_text.startswith("```json"):
                         json_text = json_text[7:]
                     elif json_text.startswith("```"):
                         json_text = json_text[3:]
                     if json_text.endswith("```"):
                         json_text = json_text[:-3]
                     json_text = json_text.strip()
                     
                     # Try to parse the JSON
                     parsed_output = None
                     json_error = None
                     
                     # First try with standard json
                     try:
                         parsed_output = json.loads(json_text)
                         logging.info("Successfully parsed final output with standard json module.")
                     except json.JSONDecodeError as e:
                         json_error = e
                         logging.warning(f"Standard json parsing failed: {e}")
                         
                         # If json5 is available, try with that
                         if HAS_JSON5:
                             try:
                                 parsed_output = json5.loads(json_text)
                                 logging.info("Successfully parsed final output with json5 module.")
                             except Exception as json5_error:
                                 logging.error(f"Both json and json5 parsing failed. Last error: {json5_error}")
                                 # Re-raise the original error if both fail
                                 raise json_error
                         else:
                             # Re-raise the original error if json5 isn't available
                             raise
                         
                     # Process the parsed output
                     if parsed_output:
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
                 else:
                     logging.error("Could not find 'Final Answer:' marker in the output")
                     logging.warning("Agent output does not contain the expected Final Answer marker.")
                     
                     # As a fallback, try to find any JSON-like structure in the output
                     if HAS_JSON5:
                         logging.info("Attempting to find and parse any JSON-like structure in the output...")
                         
                         # Look for chunks that might be JSON
                         curly_start = final_output_str.find('{')
                         curly_end = final_output_str.rfind('}')
                         
                         if curly_start != -1 and curly_end != -1 and curly_end > curly_start:
                             potential_json = final_output_str[curly_start:curly_end+1]
                             try:
                                 parsed_output = json5.loads(potential_json)
                                 logging.info("Found and parsed JSON-like structure with json5.")
                                 
                                 # Process the parsed output (same as above)
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
                             except Exception as e:
                                 logging.error(f"Fallback JSON parsing failed: {e}")

             except json.JSONDecodeError as json_error:
                 logging.error(f"Failed to parse final output string as JSON: {json_error}")
                 logging.warning("Agent did not return the expected JSON format in 'output'. The raw string was logged above.")
             except Exception as parse_exc:
                 logging.error(f"An unexpected error occurred during output parsing: {parse_exc}")
                 logging.debug(traceback.format_exc())  # More detailed traceback
        else:
             logging.warning("Agent response format unexpected or missing 'output' key.")
             logging.info(f"Full Response: {response}")


    except Exception as e:
        logging.error(f"An error occurred during agent execution: {e}", exc_info=True)

    logging.info("Experiment 2 Run Finished.")
    return log_filepath # Return log path for evaluation scripts

if __name__ == "__main__":
    run_experiment() 