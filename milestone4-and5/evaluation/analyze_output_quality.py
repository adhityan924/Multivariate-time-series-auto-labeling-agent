# Evaluation Script for Metric D: Output Quality (Explanation & Uncertainty)

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

def parse_explanation_uncertainty(log_filepath: str) -> tuple[str | None, str | None]:
    """
    Parses the agent's log file to extract the final 'explanation' and 'uncertainty_notes'.
    Tries multiple approaches to find the output in different log formats.
    """
    explanation = None
    uncertainty = None
    try:
        # Add errors='ignore' to handle potential non-UTF8 characters
        with open(log_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

            # Approach 1: Look for JSON output directly
            json_content = None
            for i, line in enumerate(lines):
                line = line.strip()
                # Look for JSON content that contains our fields of interest
                if '"explanation":' in line and '"uncertainty_notes":' in line:
                    # Try to extract the JSON object
                    try:
                        # Find the start of the JSON object
                        json_start = line.find('{')
                        if json_start >= 0:
                            json_content = line[json_start:]
                            # Parse the JSON
                            parsed_json = json.loads(json_content)
                            if 'explanation' in parsed_json and 'uncertainty_notes' in parsed_json:
                                explanation = parsed_json['explanation']
                                uncertainty = parsed_json['uncertainty_notes']
                                logging.info(f"Successfully extracted explanation and uncertainty from JSON in line {i+1}")
                                break
                    except json.JSONDecodeError:
                        # If JSON parsing fails, continue to the next line
                        continue

            # Approach 2: Look for specific log markers
            if explanation is None or uncertainty is None:
                in_final_output_section = False
                json_block_started = False
                json_content = ""

                for i, line in enumerate(lines):
                    line = line.strip()

                    # Check for markers that indicate we're in the final output section
                    if "Successfully parsed final output JSON" in line or "Agent Final Output" in line:
                        in_final_output_section = True
                        continue

                    # If we're in the final output section, look for the start of a JSON block
                    if in_final_output_section and line.startswith("{") and not json_block_started:
                        json_block_started = True
                        json_content = line
                        continue

                    # If we're collecting a JSON block, add the line to our content
                    if json_block_started:
                        json_content += line
                        # Check if this line might be the end of the JSON block
                        if line.endswith("}"):
                            # Try to parse the collected JSON
                            try:
                                parsed_json = json.loads(json_content)
                                if 'explanation' in parsed_json and 'uncertainty_notes' in parsed_json:
                                    explanation = parsed_json['explanation']
                                    uncertainty = parsed_json['uncertainty_notes']
                                    logging.info(f"Successfully extracted explanation and uncertainty from JSON block")
                                    break
                            except json.JSONDecodeError:
                                # If JSON parsing fails, continue collecting lines
                                continue

            # Approach 3: Look for specific log lines with the explanation and uncertainty
            if explanation is None or uncertainty is None:
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line.startswith("Explanation:") or "Explanation: " in line:
                        explanation = line.split("Explanation:", 1)[1].strip()
                    elif line.startswith("Uncertainty Notes:") or "Uncertainty Notes: " in line:
                        uncertainty = line.split("Uncertainty Notes:", 1)[1].strip()

                    # Check if we found both
                    if explanation is not None and uncertainty is not None:
                        logging.info(f"Successfully extracted explanation and uncertainty from log lines")
                        break

            # Approach 4: Last resort - look for any lines that might contain our data
            if explanation is None or uncertainty is None:
                for i, line in enumerate(lines):
                    if "explanation" in line.lower() and explanation is None:
                        # Try to extract explanation from this line
                        try:
                            # Look for text after "explanation"
                            parts = line.lower().split("explanation", 1)
                            if len(parts) > 1:
                                # Find the first quote after "explanation"
                                quote_pos = parts[1].find('"')
                                if quote_pos >= 0:
                                    # Find the closing quote
                                    start_pos = quote_pos + 1
                                    end_pos = parts[1].find('"', start_pos)
                                    if end_pos > start_pos:
                                        explanation = parts[1][start_pos:end_pos]
                                        logging.info(f"Extracted explanation using fallback method")
                        except Exception as e:
                            logging.warning(f"Failed to extract explanation from line {i+1}: {e}")

                    if "uncertainty" in line.lower() and uncertainty is None:
                        # Try to extract uncertainty from this line
                        try:
                            # Look for text after "uncertainty"
                            parts = line.lower().split("uncertainty", 1)
                            if len(parts) > 1:
                                # Find the first quote after "uncertainty"
                                quote_pos = parts[1].find('"')
                                if quote_pos >= 0:
                                    # Find the closing quote
                                    start_pos = quote_pos + 1
                                    end_pos = parts[1].find('"', start_pos)
                                    if end_pos > start_pos:
                                        uncertainty = parts[1][start_pos:end_pos]
                                        logging.info(f"Extracted uncertainty using fallback method")
                        except Exception as e:
                            logging.warning(f"Failed to extract uncertainty from line {i+1}: {e}")

            # Log the results of our parsing attempts
            if explanation is None and uncertainty is None:
                logging.warning("Could not find explanation or uncertainty notes in the log file.")
            elif explanation is None:
                logging.warning("Found uncertainty notes but could not find explanation in the log file.")
            elif uncertainty is None:
                logging.warning("Found explanation but could not find uncertainty notes in the log file.")
            else:
                logging.info(f"Successfully parsed explanation and uncertainty from {log_filepath}")

        return explanation, uncertainty
    except FileNotFoundError:
        logging.error(f"Log file not found: {log_filepath}")
        return None, None
    except Exception as e:
        logging.error(f"Error parsing log file {log_filepath}: {e}")
        return None, None

def evaluate_output_quality(explanation: str | None, uncertainty: str | None) -> str:
    """Uses an LLM-as-Judge to evaluate the quality of the explanation and uncertainty notes."""
    if not evaluator_llm:
        return "Error: Evaluator LLM not available."
    if explanation is None and uncertainty is None:
        # Generate a minimal placeholder for evaluation
        logging.warning("Empty explanation and uncertainty detected. Using placeholder for evaluation.")
        explanation = "The agent identified segments similar to the input pattern based on statistical properties."
        uncertainty = "There may be some uncertainty in the pattern identification process."
        # Return a warning message instead of proceeding with evaluation
        return "Warning: Explanation and uncertainty notes were empty. Please check log file format or agent execution."

    # Handle cases where one might be missing
    explanation_text = explanation if explanation is not None else "(Not provided)"
    uncertainty_text = uncertainty if uncertainty is not None else "(Not provided)"

    # Define the prompt for the LLM-as-Judge
    judge_prompt_template = """
You are an expert evaluator assessing the output quality of a time-series analysis agent.
Analyze the provided Explanation and Uncertainty Notes based on the following criteria. Provide ratings (1-5, 5 being best) and brief justifications.

**Evaluation Criteria:**
1.  **Explanation Coherence & Clarity:** Is the explanation easy to understand? Does it logically connect the findings to the task?
2.  **Explanation Accuracy & Relevance:** Does the explanation accurately reflect the likely analysis process (based on typical time-series tasks)? Is it relevant to the goal of finding similar segments?
3.  **Uncertainty Identification:** Did the agent appropriately identify and articulate any uncertainties, limitations, or borderline cases? Is the level of confidence conveyed reasonable? (Rate lower if no uncertainty notes were provided but some uncertainty would be expected).
4.  **Helpfulness:** Overall, how helpful are the explanation and uncertainty notes for a user trying to understand the agent's results?

**Agent Output:**
Explanation:
{explanation}

Uncertainty Notes:
{uncertainty}

**Your Evaluation:**
Provide your assessment in JSON format:
```json
{{
  "explanation_coherence_clarity": {{ "rating": <1-5>, "justification": "..." }},
  "explanation_accuracy_relevance": {{ "rating": <1-5>, "justification": "..." }},
  "uncertainty_identification": {{ "rating": <1-5>, "justification": "..." }},
  "helpfulness": {{ "rating": <1-5>, "justification": "..." }},
  "overall_comment": "Brief overall assessment of output quality."
}}
```
"""
    judge_prompt = judge_prompt_template.format(explanation=explanation_text, uncertainty=uncertainty_text)

    logging.info("Invoking LLM-as-Judge for output quality...")
    try:
        response = evaluator_llm.invoke(judge_prompt)
        evaluation_content = response.content # Assuming response.content holds the text
        logging.info("LLM-as-Judge evaluation received.")

        # Basic parsing attempt for the JSON within the response
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', evaluation_content, re.DOTALL)
            if json_match:
                evaluation_json_str = json_match.group(1)
                parsed_evaluation = json.loads(evaluation_json_str)
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
    parser = argparse.ArgumentParser(description="Analyze Agent Output Quality (Metric D)")
    parser.add_argument("log_filepath", help="Path to the agent's log file containing the output.")
    args = parser.parse_args()

    logging.info(f"Processing log file: {args.log_filepath}")

    # 1. Parse Explanation and Uncertainty
    explanation, uncertainty = parse_explanation_uncertainty(args.log_filepath)

    # 2. Evaluate Output Quality using LLM-as-Judge
    quality_evaluation = evaluate_output_quality(explanation, uncertainty)

    # 3. Output Results
    print("\n--- Output Quality Metrics (Metric D) ---")
    print("Parsed Values:")
    print(f"  - Explanation: {explanation if explanation else '(Not found)'}")
    print(f"  - Uncertainty Notes: {uncertainty if uncertainty else '(Not found)'}")
    print("\nQuality Assessment (LLM-as-Judge):")
    print(quality_evaluation)
    print("----------------------------------------")

if __name__ == "__main__":
    main()
