# LangChain Experiment 1: Retrieval-Augmented Time-Series Labeling Agent

This project implements a LangChain agent designed to analyze time-series data, identify segments similar to a given pattern using domain knowledge, and label them appropriately. It utilizes Retrieval-Augmented Generation (RAG) with ChromaDB for domain knowledge and an OpenAI Tools agent for execution.

## Project Structure

```
langchain-experiment1/
├── data/                     # Placeholder for input time-series CSV data
│   └── your_timeseries.csv   # Example placeholder - replace with actual data
├── db/                       # Persistent storage for ChromaDB vector store (created at runtime)
├── evaluation/               # Scripts for evaluating agent performance
│   ├── analyze_output_quality.py
│   ├── analyze_trajectory.py
│   ├── calculate_similarity.py
│   └── calculate_stability.py
├── experiment_1/             # Python virtual environment directory
├── knowledge/                # Domain knowledge source files
│   └── paper.pdf             # Example PDF document for RAG
├── logs/                     # Directory for storing execution logs (created at runtime)
├── src/                      # Source code for the agent and tools
│   ├── agent_setup.py        # Configures the LLM, tools, prompt, and agent executor
│   └── tools.py              # Defines LangChain tools for data loading, RAG, and analysis
├── config.py                 # Configuration variables (file paths, query params, API key loading)
├── requirements.txt          # Python dependencies
├── run_experiment_1.py       # Main script to execute the agent
└── tasks-cline.md            # Breakdown of implementation tasks (completed)
```

## Setup

1.  **Clone/Download:** Obtain the project files.
2.  **Create Virtual Environment:**
    *   Navigate to the `langchain-experiment1` directory.
    *   Create a Python virtual environment (e.g., using `venv`):
        ```bash
        python -m venv experiment_1
        ```
3.  **Activate Virtual Environment:**
    *   Windows (Command Prompt/PowerShell): `.\experiment_1\Scripts\activate`
    *   macOS/Linux: `source experiment_1/bin/activate`
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **API Key:**
    *   Ensure your OpenAI API key is set as an environment variable named `OPENAI_API_KEY`.
    *   Alternatively, create a file named `.env` inside the `langchain-experiment1` directory and add the line: `OPENAI_API_KEY='your_actual_api_key'`
6.  **Dataset:**
    *   Place your time-series data file (in CSV format) inside the `data/` directory.
    *   Update the `DATASET_PATH` variable in `config.py` to point to your file (e.g., `DATASET_PATH = "data/my_data.csv"`).
7.  **Knowledge Base:**
    *   Place the relevant domain knowledge PDF file inside the `knowledge/` directory.
    *   Ensure the `KNOWLEDGE_PDF_PATH` variable in `config.py` points to this file (default is `knowledge/paper.pdf`).
8.  **Query Parameters:**
    *   Review and adjust the example query parameters (`QUERY_START_ROW`, `QUERY_END_ROW`, `QUERY_COLUMN_NAME`, `QUERY_LABEL`) in `config.py` to match your analysis target.

## Running the Experiment

1.  Ensure your virtual environment (`experiment_1`) is activated.
2.  Navigate to the `langchain-experiment1` directory.
3.  Run the main script:
    ```bash
    python run_experiment_1.py
    ```
    *   The script will load data, initialize the vector store (embedding the PDF on the first run), invoke the agent, and log the final output.
    *   Detailed execution logs, including the agent's trace (captured via `LoggingCallbackHandler`), will be saved to a timestamped file in the `logs/` directory.

## Running Evaluations

1.  Ensure the experiment has been run at least once to generate log files in the `logs/` directory. For Metric C (Stability), run the experiment multiple times.
2.  Ensure your virtual environment is activated.
3.  Navigate to the `langchain-experiment1` directory.
4.  Run the desired evaluation script, passing the relevant log file path(s) as arguments:

    *   **Metric A (Similarity):**
        ```bash
        python evaluation/calculate_similarity.py logs/<log_file_name>.log
        ```
    *   **Metric B (Trajectory Analysis):**
        ```bash
        python evaluation/analyze_trajectory.py logs/<log_file_name>.log
        ```
    *   **Metric C (Stability):**
        ```bash
        python evaluation/calculate_stability.py logs/<log_file_1>.log logs/<log_file_2>.log ...
        ```
    *   **Metric D (Output Quality):**
        ```bash
        python evaluation/analyze_output_quality.py logs/<log_file_name>.log
        ```

## Technical Implementation Details

*   **Agent Type:** Uses `create_openai_tools_agent` which leverages OpenAI's native tool calling feature for more robust argument handling compared to standard ReAct text parsing.
*   **LLM:** Configured in `src/agent_setup.py` to use `ChatOpenAI` (currently set to `o3-mini-2025-01-31`).
*   **Embeddings:** Uses `OpenAIEmbeddings` (default model) configured in `src/tools.py`.
*   **Vector Store:** Uses `ChromaDB` for persistent storage (`db/` directory). The `setup_vector_store` function in `src/tools.py` handles initialization and embeds the knowledge PDF only if the collection is empty.
*   **Tools (`src/tools.py`):**
    *   `load_data`: Loads the CSV dataset into a global pandas DataFrame.
    *   `get_segment`: Extracts specified rows/columns from the loaded DataFrame.
    *   `calculate_basic_stats`: Calculates statistics on a provided segment string (output of `get_segment`).
    *   `query_domain_knowledge`: Performs RAG lookup against the ChromaDB vector store.
*   **Logging:** Uses Python's standard `logging` module configured in `run_experiment_1.py`. A `LoggingCallbackHandler` is passed to the `AgentExecutor` to capture detailed execution traces (LLM calls, tool usage) into the log file.
*   **Configuration (`config.py`):** Centralizes file paths, API key loading, and query parameters.
*   **Evaluation (`evaluation/`):** Provides scripts to assess different aspects of the agent's performance based on the generated logs. Metric B (`analyze_trajectory.py`) and Metric D (`analyze_output_quality.py`) use the configured LLM as a judge.
