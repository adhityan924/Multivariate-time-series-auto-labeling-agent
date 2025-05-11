import pandas as pd
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI # Using OpenAI for agent reasoning, though core logic is local
import os
from dotenv import load_dotenv
from colorama import Fore, Style

# Load environment variables (like OPENAI_API_KEY if needed for the agent's LLM)
load_dotenv()

# Define the tool for loading data
@tool
def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a specified CSV file path into a pandas DataFrame.
    Handles potential FileNotFoundError.
    """
    print(f"{Fore.CYAN}Attempting to load data from: {file_path}{Style.RESET_ALL}")
    try:
        # Explicitly use forward slashes or raw strings for Windows paths if needed,
        # but pandas usually handles mixed slashes well. Let's try directly first.
        df = pd.read_csv(file_path)
        # Basic validation (example: check if empty)
        if df.empty:
            print(f"{Fore.YELLOW}Warning: Loaded CSV file is empty: {file_path}{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}Successfully loaded data. Shape: {df.shape}{Style.RESET_ALL}")
        return df
    except FileNotFoundError:
        print(f"{Fore.RED}Error: File not found at {file_path}{Style.RESET_ALL}")
        # Return an empty DataFrame or raise an error, depending on desired handling
        return pd.DataFrame()
    except Exception as e:
        print(f"{Fore.RED}Error loading CSV file {file_path}: {e}{Style.RESET_ALL}")
        return pd.DataFrame()

# Define the prompt template for the agent
# This agent is simple, primarily executing the tool.
# The prompt guides the LLM (if used) on how to use the tool.
prompt_template = """
You are an agent responsible for loading data from a CSV file.
You have access to the following tools:
{tools}

Use the 'load_csv_data' tool (tool name: {tool_names}) to load the data from the given file path.

Input: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(prompt_template)

# Define the agent
# Note: Using ChatOpenAI here for the reasoning part of the agent framework.
# The actual data loading is local via the tool.
# Ensure OPENAI_API_KEY is set in your environment or .env file.
# If you don't have/want an OpenAI key, you might need to use a different LLM
# or a simpler agent structure if the LLM reasoning isn't strictly necessary for this simple task.
# For this example, we assume an LLM is part of the Langchain agent setup.
try:
    # Check if API key is available, otherwise use a placeholder or handle differently
    if not os.getenv("OPENAI_API_KEY"):
        print(f"{Fore.YELLOW}Warning: OPENAI_API_KEY not found. Agent reasoning might be limited or fail.{Style.RESET_ALL}")
        # Provide a dummy LLM or handle the case where no LLM is needed/available
        # For now, we proceed assuming it might be set elsewhere or the user knows.
    llm = ChatOpenAI(temperature=0) # Low temperature for deterministic behavior
    tools = [load_csv_data]
    agent = create_react_agent(llm, tools, prompt)
    data_loader_agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # Set verbose=True for debugging
except Exception as e:
    print(f"{Fore.RED}Error initializing LLM or Agent components: {e}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Ensure OPENAI_API_KEY is set in your environment or .env file if using OpenAI.{Style.RESET_ALL}")
    # Fallback or exit if agent setup fails
    data_loader_agent_executor = None

# Function to invoke the agent
def run_data_loader_agent(file_path: str) -> pd.DataFrame:
    """Invokes the data loader agent."""
    if data_loader_agent_executor is None:
        print(f"{Fore.RED}Data Loader Agent Executor not initialized. Cannot load data.{Style.RESET_ALL}")
        return pd.DataFrame()

    try:
        # The input format depends on how the prompt expects it.
        # Here, we pass the file path directly as the 'input'.
        result = data_loader_agent_executor.invoke({"input": f"Load data from the file path: {file_path}"})

        # The actual DataFrame is returned by the tool execution,
        # but the agent executor wraps it. We need to extract it.
        # The exact structure of 'result' might vary based on Langchain version/agent type.
        # Often, the final tool output is in 'output' or similar key.
        # Let's assume the tool directly returns the df and the executor passes it through.
        # A safer approach might be to modify the tool/agent to ensure the df is clearly returned.
        # For now, we'll try accessing 'output'. If it fails, direct tool call might be needed as fallback.

        # --- Refined approach: Directly call the tool for simplicity ---
        # Given the simplicity, directly calling the tool might be more robust than relying
        # on the LLM agent to parse the result correctly for just loading.
        # The agent framework is demonstrated, but practical use might simplify.
        print(f"{Fore.BLUE}Agent execution finished. Attempting direct tool call for robust DataFrame retrieval.{Style.RESET_ALL}")
        df_result = load_csv_data(file_path=file_path)
        return df_result

        # --- Original Agent Invocation Result Handling (Less Robust for direct DF return) ---
        # if isinstance(result, dict) and 'output' in result:
        #     # Check if the output is likely a DataFrame (this is tricky without knowing the exact agent return format)
        #     # This part is fragile and depends heavily on the agent's internal processing.
        #     # It might return a string description instead of the DataFrame object.
        #     print(f"{Fore.YELLOW}Agent result['output']: {result['output']}. Attempting to interpret as DataFrame.{Style.RESET_ALL}")
        #     # Heuristic: If the output is a string confirming success, we might need another way
        #     # to get the actual DataFrame (e.g., storing it in a shared context, which adds complexity).
        #     # Let's assume for now the agent somehow returns the DF object directly in 'output'
        #     if isinstance(result['output'], pd.DataFrame):
        #          return result['output']
        #     else:
        #         print(f"{Fore.RED}Agent did not return a DataFrame directly in 'output'. Returning empty DataFrame.{Style.RESET_ALL}")
        #         return pd.DataFrame() # Fallback
        # else:
        #     print(f"{Fore.RED}Unexpected agent result format: {result}. Returning empty DataFrame.{Style.RESET_ALL}")
        #     return pd.DataFrame()

    except Exception as e:
        print(f"{Fore.RED}Error running data loader agent: {e}{Style.RESET_ALL}")
        return pd.DataFrame()
