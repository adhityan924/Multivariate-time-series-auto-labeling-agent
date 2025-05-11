# This file will contain the agent configuration, including LLM, tools, and prompt.
import os
from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI # No longer needed
from langchain_openai import ChatOpenAI # Added
# Import tools from top-level src directory
import sys
import os
# Add the parent directory (which contains the top-level src) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.tools import load_data, get_segment, calculate_basic_stats, query_domain_knowledge, setup_vector_store
from langchain.prompts import PromptTemplate
# from langchain.agents import create_react_agent # No longer needed
from langchain.agents import AgentExecutor, create_openai_tools_agent # Ensure this is the correct import

# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY") # Changed from GEMINI_API_KEY
if not API_KEY:
    print("Error: OPENAI_API_KEY not found in environment variables. Agent setup requires the API key.") # Changed
    # In a real application, you might raise an error or exit
    # For now, we'll proceed but LLM instantiation will likely fail later if key is truly missing

# --- Agent Setup Components ---

llm = None
tools = []
react_prompt_template = None
agent = None
agent_executor = None

# --- 5.1. Instantiate LLM ---
try:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", # Changed from o3-mini-2025-01-31 to gpt-3.5-turbo
        openai_api_key=API_KEY,
        temperature=0.2 # Added temperature for more deterministic outputs
    )
    print("LLM (ChatOpenAI) instantiated successfully.")
except Exception as e:
    print(f"Error instantiating LLM: {e}")
    llm = None # Ensure llm is None if instantiation fails

# --- 5.2. Assemble Tools ---
# Ensure the vector store is set up before assembling tools that might need it
# (query_domain_knowledge implicitly calls setup_vector_store if needed)
# setup_vector_store() # Optionally call here, or let the tool handle it lazily

tools = [
    load_data,
    get_segment,
    calculate_basic_stats,
    query_domain_knowledge
]
print(f"Assembled tools: {[tool.name for tool in tools]}")


# --- 5.3. Define Prompt Template ---

# Define the core ReAct prompt structure
# Note: Tool descriptions are automatically added by the agent creation process.
react_prompt_str = """
You are a specialized time-series analysis agent. Your goal is to identify segments in a loaded time-series dataset that are similar to a given input segment and label them appropriately.

**Dataset Context:**
The dataset has been loaded. You can use tools to interact with it.

**Input Query Details:**
The user wants to find segments similar to the one defined by:
- Start Row: {input_start_row}
- End Row: {input_end_row}
- Column: {input_column_name}
The label to assign to similar segments is: {input_label}

**Domain Constraints & Assumptions:**
- [Placeholder: Insert specific constraints derived from the domain knowledge PDF here. e.g., "Focus on patterns where the value increases rapidly then stabilizes."]
- Only label segments that closely match the pattern defined by the input segment.
- Do not label other patterns, even if they seem interesting.

**Task:**
1.  Analyze the input segment (rows {input_start_row} to {input_end_row}, column '{input_column_name}') using the available tools (e.g., get_segment, calculate_basic_stats).
2.  If necessary, query the domain knowledge base for specific characteristics or definitions related to the pattern or label '{input_label}'.
3.  Search the rest of the dataset for segments exhibiting similar characteristics. You might need to iterate using `get_segment` and `calculate_basic_stats`.
4.  Compile a list of identified similar segments.
5.  Provide a clear explanation for why these segments were identified, referencing specific statistical properties or knowledge base insights.
6.  Flag any uncertainties or borderline cases.

**Instructions for Response Format:**
Think step-by-step about how to achieve the task using your available tools.
When you have enough information, provide the final answer directly in the required JSON format specified below. Your intermediate thoughts and tool usage will be processed automatically.

**Required Final Output Format:**
Your final answer (inside the "Final Answer" action_input) MUST be a JSON object string with the following structure:
```json
{{
  "identified_segments": [
    {{"start_row": <start_row_int>, "end_row": <end_row_int>}},
    ...
  ],
  "assigned_label": "{input_label}",
  "explanation": "Detailed explanation of the findings, referencing statistics, knowledge base info, and similarity criteria.",
  "uncertainty_notes": "Notes on any uncertainties, borderline cases, or limitations."
}}
```

**Begin!**

Input Query: {input}
Agent Scratchpad: {agent_scratchpad}
"""

# Create the PromptTemplate object
try:
    react_prompt_template = PromptTemplate.from_template(react_prompt_str)
    # Add partial variables that are known now (like tool descriptions)
    # Note: create_react_agent typically handles formatting tools into the prompt,
    # so explicitly adding {tools} and {tool_names} might be redundant depending on its implementation.
    # However, defining it here makes the structure clear. We might need to adjust if create_react_agent duplicates this.
    # react_prompt_template = react_prompt_template.partial(
    #     tools=render_text_description(tools), # Function to format tool descriptions
    #     tool_names=", ".join([t.name for t in tools]),
    # )
    print("ReAct prompt template created.")
except Exception as e:
    print(f"Error creating prompt template: {e}")
    react_prompt_template = None

# --- 6.1. Create Agent Logic ---
# Ensure LLM, tools, and prompt are available before creating the agent
if llm and tools and react_prompt_template:
    try:
        # Use create_openai_tools_agent instead of create_react_agent
        agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=react_prompt_template)
        print("OpenAI Tools Agent created successfully.")
    except Exception as e:
        print(f"Error creating OpenAI Tools agent: {e}")
        agent = None
else:
    print("Error: LLM, tools, or prompt not initialized correctly. Cannot create agent.")
    agent = None

# --- 6.2. Initialize AgentExecutor ---
if agent and tools:
    try:
        # Initialize the AgentExecutor with the agent and tools
        # Set verbose=True for detailed logging of the agent's thought process
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors="Agent failed to parse output. Please check the thought process and final output format." # Provide specific error message
        )
        print("AgentExecutor initialized successfully.")
    except Exception as e:
        print(f"Error initializing AgentExecutor: {e}")
        agent_executor = None
else:
    print("Error: Agent not created or tools not available. Cannot initialize AgentExecutor.")
    agent_executor = None

print("agent_setup.py loaded and agent components initialized (if possible).")
