# This file will contain the Plan-and-Solve agent configuration, including LLM, tools, and prompt.
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# Import tools from top-level src directory
import sys
# Add the parent directory (which contains the top-level src) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.tools import load_data, get_segment, calculate_basic_stats, query_domain_knowledge, setup_vector_store
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent

# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("Error: OPENAI_API_KEY not found in environment variables. Agent setup requires the API key.")
    # In a real application, you might raise an error or exit
    # For now, we'll proceed but LLM instantiation will likely fail later if key is truly missing

# --- Agent Setup Components ---

llm = None
tools = []
ps_prompt_template = None
agent = None
agent_executor = None

# --- 5.1. Instantiate LLM ---
# Using the same model as experiment1
try:
    llm = ChatOpenAI(
        model="o3-mini-2025-01-31",
        openai_api_key=API_KEY
    )
    print("LLM (ChatOpenAI) instantiated successfully.")
except Exception as e:
    print(f"Error instantiating LLM: {e}")
    llm = None # Ensure llm is None if instantiation fails

# --- 5.2. Assemble Tools ---
# Using the same tools as experiment1
tools = [
    load_data,
    get_segment,
    calculate_basic_stats,
    query_domain_knowledge
]
print(f"Assembled tools: {[tool.name for tool in tools]}")

# --- 7. Define Plan-and-Solve Prompt ---

# Define the Plan-and-Solve prompt structure with explicit planning phase
ps_prompt_str = """
You are a specialized time-series analysis agent. Your goal is to identify segments in a loaded time-series dataset that are similar to a given input segment and label them appropriately.

**Dataset Context:**
The dataset has been loaded. You can use tools to interact with it.

**Input Query Details:**
The user wants to find segments similar to the one defined by:
- Start Row: {input_start_row}
- End Row: {input_end_row}
- Column: {input_column_name}
The label to assign to similar segments is: {input_label}

**Task:**
Identify segments in the time-series data that are similar to the input segment and label them with "{input_label}".

**Available Tools:**
You have access to tools for:
1. Loading data (`load_data`)
2. Getting specific data segments (`get_segment`)
3. Calculating basic statistics (`calculate_basic_stats`)
4. Querying domain knowledge (`query_domain_knowledge`)

**IMPORTANT - PLAN-AND-SOLVE APPROACH:**
You must use a plan-and-solve approach with these specific phases:

**PHASE 1: PLANNING**
Before taking any action, develop a detailed plan with these components:

1. **Initial Problem Analysis:**
   - Analyze what the time-series labeling task requires
   - Identify what information and insights you'll need
   - Clarify any assumptions about the data

2. **Task Decomposition:**
   - Break down the labeling problem into clear sub-tasks
   - Establish dependencies between sub-tasks
   - Define clear success criteria for each sub-task

3. **Strategy Formulation:**
   - Select appropriate analysis techniques
   - Decide which tools to use for each sub-task
   - Identify potential challenges and mitigation strategies

4. **Execution Plan:**
   - Create a step-by-step sequence of actions
   - Specify which tools will be used at each step
   - Establish how to validate results at each step

5. **Reflection on Plan:**
   - Assess if the plan covers all requirements
   - Identify potential weaknesses or failure points
   - Consider alternative approaches if initial plan fails

**PHASE 2: EXECUTION**
After completing the planning phase, follow these steps:

1. Execute your plan step by step
2. Document your observations and findings at each step
3. Adjust your approach based on what you learn
4. Validate your results against your success criteria
5. Prepare the final output with identified segments

**Required Final Output Format:**
Your final answer MUST begin with the exact text "Final Answer:" followed by a JSON object with the following structure:
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

VERY IMPORTANT: Your final response must start with "Final Answer:" followed by the JSON. Without this exact prefix, your answer cannot be properly processed.

**Begin!**

Input Query: {input}
Agent Scratchpad: {agent_scratchpad}
"""

# Create the Plan-and-Solve PromptTemplate object
try:
    ps_prompt_template = PromptTemplate.from_template(ps_prompt_str)
    print("Plan-and-Solve prompt template created.")
except Exception as e:
    print(f"Error creating Plan-and-Solve prompt template: {e}")
    ps_prompt_template = None

# --- 6.1. Create Agent Logic ---
# Note: The actual prompt template will be defined in Task 7
def create_plan_and_solve_agent(ps_prompt_template):
    """
    Create a Plan-and-Solve agent using the OpenAI Tools agent format.
    This function should be called after the ps_prompt_template is defined.
    """
    if llm and tools and ps_prompt_template:
        try:
            # Use create_openai_tools_agent with the Plan-and-Solve prompt
            agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=ps_prompt_template)
            print("Plan-and-Solve OpenAI Tools Agent created successfully.")
            return agent
        except Exception as e:
            print(f"Error creating Plan-and-Solve agent: {e}")
            return None
    else:
        print("Error: LLM, tools, or prompt not initialized correctly. Cannot create agent.")
        return None

# --- 6.2. Initialize AgentExecutor ---
def create_agent_executor(agent):
    """
    Initialize the AgentExecutor with the given agent and tools.
    This function should be called after the agent is created.
    """
    if agent and tools:
        try:
            # Initialize the AgentExecutor with the agent and tools
            # Same configuration as experiment1 for consistent comparison
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors="Agent failed to parse output. Please check the thought process and final output format."
            )
            print("AgentExecutor initialized successfully.")
            return agent_executor
        except Exception as e:
            print(f"Error initializing AgentExecutor: {e}")
            return None
    else:
        print("Error: Agent not created or tools not available. Cannot initialize AgentExecutor.")
        return None

print("agent_setup.py for Plan-and-Solve agent loaded and LLM configured.") 