import numpy as np
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from colorama import Fore, Style
from typing import List

# Assuming utils are in ../utils relative to agents directory
from src.utils.embedding_utils import compute_embedding

# Load environment variables
load_dotenv()

# Define the tool for embedding
@tool
def generate_chunk_embedding(chunk_data: List[float]) -> List[float]:
    """
    Takes a list of numerical data (representing a time series chunk),
    converts it to a numpy array, and computes its vector embedding
    using FFT and time-domain features.
    Returns the embedding as a list of floats.
    """
    # Note: The tool input is defined as List[float] for compatibility with agent invocation,
    # but the underlying compute_embedding function expects a numpy array.
    print(f"{Fore.CYAN}Attempting to generate embedding for chunk (size: {len(chunk_data)}).{Style.RESET_ALL}")
    try:
        chunk_array = np.array(chunk_data)
        if chunk_array.ndim != 1:
            raise ValueError("Input chunk data must be 1-dimensional")

        embedding_vector = compute_embedding(chunk_array)

        if not embedding_vector: # Check if embedding computation returned empty list (error)
             print(f"{Fore.YELLOW}Embedding computation failed or returned empty for the chunk.{Style.RESET_ALL}")
             return []

        print(f"{Fore.GREEN}Successfully generated embedding (dimension: {len(embedding_vector)}).{Style.RESET_ALL}")
        return embedding_vector
    except ValueError as ve:
        print(f"{Fore.RED}Error during embedding pre-processing: {ve}{Style.RESET_ALL}")
        return []
    except Exception as e:
        print(f"{Fore.RED}Error generating embedding: {e}{Style.RESET_ALL}")
        return []

# Define the prompt template for the agent
prompt_template = """
You are an agent responsible for generating vector embeddings for time series chunks.
You have access to the following tools:
{tools}

Use the 'generate_chunk_embedding' tool (tool name: {tool_names}) to generate the embedding vector. It expects 'chunk_data' as a list of numbers.

Input: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(prompt_template)

# Define the agent
try:
    if not os.getenv("OPENAI_API_KEY"):
        print(f"{Fore.YELLOW}Warning: OPENAI_API_KEY not found. Agent reasoning might be limited or fail.{Style.RESET_ALL}")
    llm = ChatOpenAI(temperature=0)
    tools = [generate_chunk_embedding]
    agent = create_react_agent(llm, tools, prompt)
    embedding_agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
except Exception as e:
    print(f"{Fore.RED}Error initializing LLM or Agent components for EmbeddingAgent: {e}{Style.RESET_ALL}")
    embedding_agent_executor = None

# Function to invoke the agent
def run_embedding_agent(chunk_data: np.ndarray) -> List[float]:
    """Invokes the embedding agent for a given numpy array chunk."""
    if embedding_agent_executor is None:
        print(f"{Fore.RED}Embedding Agent Executor not initialized. Cannot perform embedding.{Style.RESET_ALL}")
        return []

    # Convert numpy array to list for the tool input
    data_list = chunk_data.tolist()

    try:
        # Similar to the chunking agent, passing large lists via ReAct prompt is inefficient.
        # We'll use the simplified approach: Agent conceptually approves/guides,
        # but the main script calls the tool function directly.

        print(f"{Fore.BLUE}Requesting Embedding Agent to process chunk (size: {len(data_list)}).{Style.RESET_ALL}")

        # Direct tool call
        print(f"{Fore.BLUE}Proceeding with direct tool call for embedding.{Style.RESET_ALL}")
        embedding = generate_chunk_embedding(chunk_data=data_list)
        return embedding

        # --- Original ReAct Invocation (Less Robust) ---
        # input_description = f"Generate embedding for the chunk data: {data_list}"
        # result = embedding_agent_executor.invoke({"input": input_description})
        #
        # # Extracting results (list of floats) from ReAct agent
        # if isinstance(result, dict) and 'intermediate_steps' in result and result['intermediate_steps']:
        #     for action, observation in result['intermediate_steps']:
        #         if action.tool == 'generate_chunk_embedding':
        #             print(f"{Fore.GREEN}Extracted embedding from agent's intermediate steps.{Style.RESET_ALL}")
        #             # Ensure observation is a list of floats
        #             if isinstance(observation, list) and all(isinstance(x, float) for x in observation):
        #                 return observation
        #             else:
        #                 print(f"{Fore.YELLOW}Tool observation is not a list of floats: {observation}{Style.RESET_ALL}")
        #                 return []
        #     print(f"{Fore.YELLOW}Could not find 'generate_chunk_embedding' output in intermediate steps.{Style.RESET_ALL}")
        #     return []
        # elif isinstance(result, dict) and 'output' in result:
        #      print(f"{Fore.YELLOW}Agent final output: {result['output']}. Attempting to interpret (might fail).{Style.RESET_ALL}")
        #      # Basic interpretation attempt (unreliable)
        #      try:
        #          # eval_result = eval(result['output']) if isinstance(result['output'], str) else result['output']
        #          # if isinstance(eval_result, list) and all(isinstance(x, float) for x in eval_result):
        #          #     return eval_result
        #          return [] # Safer fallback
        #      except:
        #          return []
        # else:
        #     print(f"{Fore.RED}Unexpected agent result format: {result}. Cannot extract embedding.{Style.RESET_ALL}")
        #     return []

    except Exception as e:
        print(f"{Fore.RED}Error running embedding agent or processing its result: {e}{Style.RESET_ALL}")
        return []
