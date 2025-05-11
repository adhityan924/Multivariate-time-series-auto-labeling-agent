import pandas as pd
import numpy as np
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from colorama import Fore, Style
from typing import List, Dict, Any

# Assuming utils are in ../utils relative to agents directory
from src.utils.chunking_utils import create_overlapping_chunks

# Load environment variables
load_dotenv()

# Define the tool for chunking
@tool
def perform_chunking(column_data: List[float], chunk_size: int, overlap: int) -> List[np.ndarray]:
    """
    Takes a list of numerical data (representing a time series column),
    a chunk size, and an overlap, and returns a list of overlapping numpy array chunks.
    Handles potential errors during chunking.
    """
    print(f"{Fore.CYAN}Attempting to chunk data. Chunk Size: {chunk_size}, Overlap: {overlap}{Style.RESET_ALL}")
    try:
        # Convert list to numpy array for the utility function
        data_array = np.array(column_data)
        if data_array.ndim != 1:
            raise ValueError("Input data must be 1-dimensional")

        chunks_generator = create_overlapping_chunks(data_array, chunk_size, overlap)
        chunks_list = list(chunks_generator) # Convert generator to list

        print(f"{Fore.GREEN}Successfully created {len(chunks_list)} chunks.{Style.RESET_ALL}")
        return chunks_list
    except ValueError as ve:
        print(f"{Fore.RED}Error during chunking pre-processing: {ve}{Style.RESET_ALL}")
        return []
    except Exception as e:
        print(f"{Fore.RED}Error during chunking: {e}{Style.RESET_ALL}")
        return []

# Define the prompt template for the agent
prompt_template = """
You are an agent responsible for splitting time series data into overlapping chunks.
You have access to the following tools:
{tools}

Use the 'perform_chunking' tool (tool name: {tool_names}) to generate the chunks. It expects 'column_data' (list of numbers), 'chunk_size' (integer), and 'overlap' (integer).

Input: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(prompt_template)

# Define the agent
try:
    if not os.getenv("OPENAI_API_KEY"):
        print(f"{Fore.YELLOW}Warning: OPENAI_API_KEY not found. Agent reasoning might be limited or fail.{Style.RESET_ALL}")
    llm = ChatOpenAI(temperature=0)
    tools = [perform_chunking]
    agent = create_react_agent(llm, tools, prompt)
    # Important: Allow returning intermediate steps if the final result needs parsing
    chunking_agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
except Exception as e:
    print(f"{Fore.RED}Error initializing LLM or Agent components for ChunkingAgent: {e}{Style.RESET_ALL}")
    chunking_agent_executor = None

# Function to invoke the agent
def run_chunking_agent(column_data: pd.Series, chunk_size: int, overlap: int) -> List[np.ndarray]:
    """Invokes the chunking agent for a given pandas Series."""
    if chunking_agent_executor is None:
        print(f"{Fore.RED}Chunking Agent Executor not initialized. Cannot perform chunking.{Style.RESET_ALL}")
        return []

    # Convert Series to list for the tool input
    data_list = column_data.tolist()

    try:
        # Construct the input string for the agent
        # The agent needs to parse this to call the tool correctly.
        # Passing structured data directly might be better if using function calling agents.
        # For ReAct, we describe the task in the input.
        input_description = (
            f"Chunk the provided time series data list. "
            f"Use chunk_size={chunk_size} and overlap={overlap}. "
            f"The data list is: {data_list}" # Sending data in prompt is inefficient for large lists
        )

        # --- Refined approach: Pass structured data if possible, or simplify ---
        # ReAct agents with string inputs struggle with large data like lists.
        # A better approach for agents needing complex inputs is:
        # 1. Use Langchain's function calling agents if the LLM supports it well.
        # 2. Pass data via context/memory (more complex setup).
        # 3. Simplify: Have the agent *orchestrate* but call the tool function directly
        #    from the main script after the agent decides *to* chunk.

        # Let's use approach 3 for robustness here: The agent confirms the action,
        # but the actual chunking (tool call) happens outside the agent's direct invoke result.
        # This avoids passing large data through the LLM prompt.

        print(f"{Fore.BLUE}Requesting Chunking Agent to process column (data preview: {data_list[:5]}...).{Style.RESET_ALL}")
        # Simulate agent deciding to chunk (optional, could just call tool directly)
        # agent_decision = chunking_agent_executor.invoke({"input": f"Should I chunk data with size {chunk_size} and overlap {overlap}?"})
        # print(f"Agent decision response: {agent_decision}")
        # if "yes" in str(agent_decision.get('output', '')).lower(): # Example decision check

        # Direct tool call after conceptual "agent approval"
        print(f"{Fore.BLUE}Proceeding with direct tool call for chunking.{Style.RESET_ALL}")
        chunks = perform_chunking(column_data=data_list, chunk_size=chunk_size, overlap=overlap)
        return chunks
        # else:
        #     print(f"{Fore.YELLOW}Agent did not confirm chunking action.{Style.RESET_ALL}")
        #     return []


        # --- Original ReAct Invocation (Less Robust for large data in prompt) ---
        # result = chunking_agent_executor.invoke({"input": input_description})
        #
        # # Extracting results from ReAct agent can be complex, especially list of arrays.
        # # Intermediate steps might contain the tool output.
        # if isinstance(result, dict) and 'intermediate_steps' in result and result['intermediate_steps']:
        #     # Find the output of the perform_chunking tool
        #     for action, observation in result['intermediate_steps']:
        #         if action.tool == 'perform_chunking':
        #             print(f"{Fore.GREEN}Extracted chunks from agent's intermediate steps.{Style.RESET_ALL}")
        #             return observation # Assuming the observation is the list of np.ndarray chunks
        #     print(f"{Fore.YELLOW}Could not find 'perform_chunking' output in intermediate steps.{Style.RESET_ALL}")
        #     return []
        # elif isinstance(result, dict) and 'output' in result:
        #      # Less likely for complex types like list of arrays
        #      print(f"{Fore.YELLOW}Agent final output: {result['output']}. Attempting to interpret (might fail).{Style.RESET_ALL}")
        #      # Try a very basic parse if it's a string representation, otherwise fail
        #      try:
        #          # This is highly unreliable
        #          # eval_result = eval(result['output']) if isinstance(result['output'], str) else result['output']
        #          # if isinstance(eval_result, list): # Further checks needed
        #          #     return eval_result
        #          return [] # Safer to return empty
        #      except:
        #          return [] # Failed to interpret
        # else:
        #     print(f"{Fore.RED}Unexpected agent result format: {result}. Cannot extract chunks.{Style.RESET_ALL}")
        #     return []

    except Exception as e:
        print(f"{Fore.RED}Error running chunking agent or processing its result: {e}{Style.RESET_ALL}")
        return []
