# Evaluation Script for Metric B: Trajectory Analysis (Efficiency & Reasoning)

import argparse
import logging
import os
import re
import sys
import json

# Add project root to sys.path to allow importing config and agent_setup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables (needed for API key)
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Setup basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Instantiate a dedicated LLM for evaluation
evaluator_llm = None
if API_KEY:
    try:
        evaluator_llm = ChatOpenAI(
            model="o3-mini-2025-01-31", # Use the same model type as the agent for consistency
            openai_api_key=API_KEY
        )
        logging.info("Evaluator LLM (ChatOpenAI) instantiated successfully.")
    except Exception as e:
        logging.error(f"Error instantiating evaluator LLM: {e}")
else:
    logging.error("OPENAI_API_KEY not found. Cannot instantiate evaluator LLM.")

def parse_trajectory_and_metrics(log_filepath: str) -> tuple[str, dict]:
    """
    Parses the agent's log file (generated with LoggingCallbackHandler) to extract
    the execution trajectory and calculate metrics like LLM calls and tool usage.
    """
    tool_counts = {}
    llm_calls = 0
    in_agent_execution = False
    trajectory_string_for_judge = "" # Store relevant parts for LLM judge

    try:
        # Add errors='ignore' to handle potential non-UTF8 characters (like ANSI color codes)
        with open(log_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # First, check if this is a LangChain log file with the expected format
        has_agent_executor_markers = False
        for line in lines:
            if "[chain:AgentExecutor]" in line:
                has_agent_executor_markers = True
                break

        # If no AgentExecutor markers found, try to extract trajectory from a different format
        # This handles logs that might not have the exact chain markers we're looking for
        if not has_agent_executor_markers:
            # Look for any agent execution indicators
            for line in lines:
                if "Agent execution" in line or "AgentExecutor" in line:
                    trajectory_string_for_judge += line
                    # Count LLM calls
                    if "llm" in line.lower() or "openai" in line.lower():
                        llm_calls += 1
                    # Count Tool Usage - simplified pattern
                    for tool_name in ["get_segment", "calculate_basic_stats", "query_domain_knowledge", "load_data"]:
                        if tool_name in line:
                            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

            # If we found some content, return it
            if trajectory_string_for_judge:
                metrics = {
                    "estimated_llm_calls": llm_calls,
                    "tool_usage_counts": tool_counts,
                    "token_count": "N/A (Requires instrumentation)",
                    "execution_time_seconds": "N/A (Requires instrumentation)"
                }
                logging.info(f"Parsed trajectory using fallback method from {log_filepath}")
                return trajectory_string_for_judge.strip(), metrics

        # Standard LangChain format parsing
        for line in lines:
            # More flexible marker detection
            is_start_marker = ("[chain/start]" in line and "AgentExecutor" in line) or "Entering Chain run with input" in line
            is_end_marker = ("[chain/end]" in line and "AgentExecutor" in line) or "Exiting Chain run with output" in line

            # Process the line if we are currently inside the agent execution block
            if in_agent_execution:
                # Append message part to trajectory string
                try:
                    # Extract message after timestamp and level (e.g., "INFO - ")
                    if " - " in line:
                        log_message_part = line.split(" - ", 2)[-1]
                        trajectory_string_for_judge += log_message_part
                    else:
                        trajectory_string_for_judge += line # Fallback
                except IndexError:
                    trajectory_string_for_judge += line # Fallback

                # Count LLM calls - more flexible pattern
                if "[llm/start]" in line or "Entering LLM run" in line:
                    llm_calls += 1
                # Count Tool Usage - more flexible pattern
                tool_match = re.search(r"\[tool/start\].*tool:(\w+)\]", line)
                if not tool_match and "tool:" in line:
                    tool_match = re.search(r"tool:(\w+)", line)
                if tool_match:
                    tool_name = tool_match.group(1)
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

            # Update the state *after* processing the line content
            # Start execution block *after* the start marker line is processed (if needed)
            if is_start_marker:
                if not in_agent_execution: # Append start marker only once
                     trajectory_string_for_judge += line # Include the start line itself
                in_agent_execution = True
            # End execution block *after* the end marker line is processed
            elif is_end_marker and in_agent_execution:
                # The end marker line was already processed above if in_agent_execution was True
                in_agent_execution = False
                # No break needed, let it finish reading the file if necessary

        # If we still don't have any trajectory, try a last resort approach
        if not trajectory_string_for_judge:
            # Just include all lines that seem relevant to agent execution
            for line in lines:
                if any(keyword in line for keyword in ["agent", "tool", "llm", "chain", "prompt", "response", "output"]):
                    trajectory_string_for_judge += line
                    # Count LLM calls
                    if "llm" in line.lower():
                        llm_calls += 1
                    # Count Tool Usage - simplified pattern
                    for tool_name in ["get_segment", "calculate_basic_stats", "query_domain_knowledge", "load_data"]:
                        if tool_name in line:
                            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        metrics = {
            "estimated_llm_calls": llm_calls,
            "tool_usage_counts": tool_counts,
            # Token counts and exact timing would ideally be captured during execution
            "token_count": "N/A (Requires instrumentation)",
            "execution_time_seconds": "N/A (Requires instrumentation)"
        }
        logging.info(f"Parsed trajectory and estimated metrics from {log_filepath}")
        # Return the constructed trajectory string and metrics
        return trajectory_string_for_judge.strip(), metrics # Strip leading/trailing whitespace

    except FileNotFoundError:
        logging.error(f"Log file not found: {log_filepath}")
        return "", {"error": "Log file not found"}
    except Exception as e:
        logging.error(f"Error parsing log file {log_filepath}: {e}")
        return "", {"error": f"Parsing failed: {e}"}


def evaluate_reasoning_quality(trajectory: str) -> str:
    """Uses an LLM-as-Judge to evaluate the agent's reasoning trajectory."""
    if not evaluator_llm:
        return "Error: Evaluator LLM not available."
    if not trajectory:
        # Generate a minimal placeholder trajectory for evaluation
        logging.warning("Empty trajectory detected. Using placeholder for evaluation.")
        trajectory = """[Agent Execution]
Agent started processing the time-series data task.
Agent used tools to analyze the data segment.
Agent identified patterns in the data.
Agent completed the task successfully.
"""
        # Return a warning message instead of proceeding with evaluation
        return "Warning: Trajectory was empty. Please check log file format or agent execution."

    # Define the prompt for the LLM-as-Judge
    judge_prompt_template = """
You are an expert evaluator assessing the reasoning quality of a language model agent designed for time-series analysis.
Analyze the following agent execution trace (including LLM outputs and tool interactions) and provide ratings (1-5, 5 being best) and brief justifications for each criterion.

**Evaluation Criteria:**
1.  **Logical Coherence:** Does the agent's process flow logically from one step to the next? Are the tool calls justified by the preceding reasoning or information?
2.  **Efficiency:** Did the agent use an appropriate number of steps? Did it avoid unnecessary actions or redundant queries?
3.  **Tool Use:** Did the agent use the available tools effectively and correctly? Was the `query_domain_knowledge` tool used when appropriate?
4.  **Reasoning Traceability:** Is it clear *why* the agent made its decisions based on the logged steps and observations?
5.  **Constraint Adherence:** Did the agent seem to follow the task instructions and domain constraints mentioned in its initial prompt (assume standard constraints if not explicitly stated in the trajectory)?

**Agent Trajectory:**
```
{trajectory}
```

**Your Evaluation:**
Provide your assessment in JSON format:
```json
{{
  "logical_coherence": {{ "rating": <1-5>, "justification": "..." }},
  "efficiency": {{ "rating": <1-5>, "justification": "..." }},
  "tool_use": {{ "rating": <1-5>, "justification": "..." }},
  "reasoning_traceability": {{ "rating": <1-5>, "justification": "..." }},
  "constraint_adherence": {{ "rating": <1-5>, "justification": "..." }},
  "overall_comment": "Brief overall assessment."
}}
```
"""
    # Update judge prompt to be less ReAct specific
    judge_prompt = judge_prompt_template.format(trajectory=trajectory)


    logging.info("Invoking LLM-as-Judge for reasoning quality...")
    try:
        response = evaluator_llm.invoke(judge_prompt)
        # Assuming the response object has a 'content' attribute with the text
        evaluation_content = response.content
        logging.info("LLM-as-Judge evaluation received.")

        # Basic parsing attempt for the JSON within the response
        try:
            # Extract JSON part (assuming it's enclosed in ```json ... ```)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', evaluation_content, re.DOTALL)
            if json_match:
                evaluation_json_str = json_match.group(1)
                parsed_evaluation = json.loads(evaluation_json_str)
                # Return the pretty-printed JSON string
                return json.dumps(parsed_evaluation, indent=2)
            else:
                logging.warning("LLM-as-Judge did not return the expected JSON format. Returning raw content.")
                return evaluation_content
        except json.JSONDecodeError as json_error:
            logging.error(f"Failed to parse LLM-as-Judge response JSON: {json_error}")
            return f"Error parsing judge response: {json_error}\nRaw Response:\n{evaluation_content}"
        except Exception as parse_exc:
             logging.error(f"Unexpected error parsing LLM-as-Judge response: {parse_exc}")
             return f"Unexpected error parsing judge response: {parse_exc}\nRaw Response:\n{evaluation_content}"

    except Exception as e:
        logging.error(f"Error invoking LLM-as-Judge: {e}")
        return f"Error during LLM-as-Judge evaluation: {e}"


def main():
    parser = argparse.ArgumentParser(description="Analyze Agent Trajectory for Metrics B")
    parser.add_argument("log_filepath", help="Path to the agent's log file.")
    args = parser.parse_args()

    logging.info(f"Processing log file: {args.log_filepath}")

    # 1. Parse Trajectory and Basic Metrics
    trajectory, metrics = parse_trajectory_and_metrics(args.log_filepath)

    # 2. Evaluate Reasoning Quality using LLM-as-Judge
    reasoning_evaluation = evaluate_reasoning_quality(trajectory)

    # 3. Output Results
    print("\n--- Trajectory Analysis Metrics (Metric B) ---")
    print("Efficiency & Tool Use:")
    print(f"  - Estimated LLM Calls: {metrics.get('estimated_llm_calls', 'N/A')}")
    print(f"  - Tool Usage Counts: {metrics.get('tool_usage_counts', 'N/A')}")
    print(f"  - Token Count: {metrics.get('token_count', 'N/A')}")
    print(f"  - Execution Time (s): {metrics.get('execution_time_seconds', 'N/A')}")
    if "error" in metrics:
        print(f"  - Parsing Error: {metrics['error']}")

    print("\nReasoning Quality (LLM-as-Judge Assessment):")
    print(reasoning_evaluation)
    print("---------------------------------------------")

if __name__ == "__main__":
    main()
