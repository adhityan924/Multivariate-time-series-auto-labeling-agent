# Multivariate Time Series Similarity & Annotation Agent

This project implements a CLI-based agent application using Langchain to find similar patterns in multivariate time series data based on user queries. It loads data from a CSV, chunks it, computes embeddings (FFT + time-domain features), stores them in ChromaDB, and allows users to query with a time series snippet and an annotation label to find the most similar stored chunks.

## Features

*   **Agent-Based Workflow:** Uses Langchain agents (conceptually) to structure tasks like data loading, chunking, embedding, and searching. (Note: For robustness with data handling, the final implementation primarily uses direct function calls orchestrated by the main script).
*   **Automatic Data Loading:** Loads data from a predefined CSV file path.
*   **Column-wise Processing:** Treats specified columns of the multivariate dataset as independent time series.
*   **Overlapping Chunking:** Splits time series into overlapping chunks to capture patterns across boundaries.
*   **Custom Embeddings:** Uses FFT magnitudes combined with time-domain statistics (mean, std, min, max) for embedding generation.
*   **Vector Similarity Search:** Leverages ChromaDB for efficient local similarity search using cosine distance.
*   **User Annotation:** Allows users to associate a label with their query sequence.
*   **Interactive CLI:** Prompts users for input and displays results in a colorized, pretty-printed JSON format.
*   **Persistent Storage:** Uses ChromaDB's persistent client to store embeddings locally, avoiding reprocessing on subsequent runs.

## Project Structure

```
langchain-multi-agent-gemini2dot5/
├── langchain-multi/           # Python Virtual Environment
├── src/
│   ├── agents/
│   │   ├── data_loader_agent.py
│   │   ├── chunking_agent.py
│   │   ├── embedding_agent.py
│   │   ├── vector_store_agent.py
│   │   └── output_agent.py
│   └── utils/
│       ├── chunking_utils.py
│       └── embedding_utils.py
├── chroma_db/                 # Local ChromaDB storage (created on first run)
├── main.py                    # Main CLI application script
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## Setup (Windows 10, Python 3.12)

1.  **Clone/Download:** Get the project files onto your local machine.
2.  **Python 3.12:** Ensure you have Python 3.12 installed and added to your system's PATH.
3.  **Create Virtual Environment:**
    *   Open a terminal (like Command Prompt or PowerShell) in the project's root directory (`langchain-multi-agent-gemini2dot5`).
    *   Run: `python -m venv langchain-multi`
4.  **Activate Virtual Environment:**
    *   In PowerShell: `.\langchain-multi\Scripts\Activate.ps1`
    *   In Command Prompt: `.\langchain-multi\Scripts\activate.bat`
    *   You should see `(langchain-multi)` prefixing your terminal prompt.
5.  **Install Dependencies:**
    *   While the virtual environment is active, run: `pip install -r requirements.txt`
6.  **Environment Variables (Optional but Recommended):**
    *   The Langchain agents are configured to potentially use `ChatOpenAI` for reasoning. If you intend to leverage this or other Langchain components requiring an API key, create a file named `.env` in the project root directory.
    *   Add your OpenAI API key to the `.env` file:
        ```dotenv
        OPENAI_API_KEY="your_openai_api_key_here"
        ```
    *   **Note:** The core functionality (data loading, chunking, embedding via FFT/stats, ChromaDB search) *does not* strictly require the OpenAI API key as implemented in `main.py` (which uses direct function calls). However, setting it prevents warnings from Langchain components during initialization.
7.  **Dataset:**
    *   Ensure the dataset CSV file exists at the path specified in `main.py`:
        `D:/CMU Forms and Data/Spring 2025 sem/AI for PM/Project Info/Dataset 2 Content/#1-mini.csv`
    *   If your dataset is located elsewhere, update the `DATA_FILE_PATH` variable in `main.py`.

## Usage

1.  **Activate Environment:** Make sure your `langchain-multi` virtual environment is activated (see step 4 in Setup).
2.  **Run the Script:** Execute the main script from the project root directory:
    ```bash
    python main.py
    ```
3.  **First Run (Data Processing):**
    *   The script will detect if the ChromaDB store (`./chroma_db`) is empty.
    *   If empty, it will load the CSV, process the specified columns, create chunks, generate embeddings, and store them in ChromaDB. This might take some time depending on the dataset size. You'll see colorized status messages.
    *   If the store already contains data, it will skip this processing step.
4.  **User Input:**
    *   The script will prompt you to enter a time series array in JSON list format (e.g., `[10.1, 10.5, 10.3, 9.8]`).
    *   It will then prompt you for an annotation label (e.g., `voltage-dip`).
5.  **Similarity Search:**
    *   The script embeds your query array using the same method (FFT + stats).
    *   It searches ChromaDB for the top 3 most similar chunks based on cosine similarity.
6.  **Output:**
    *   The results are displayed in a pretty-printed JSON format:
        ```json
        {
          "your_annotation_label": [
            [similar_chunk_1_value, ...],
            [similar_chunk_2_value, ...],
            [similar_chunk_3_value, ...]
          ]
        }
        ```
7.  **Repeat or Exit:** The script will ask if you want to perform another search (`y/n`). Enter `y` to continue or `n` to exit.

## Configuration Parameters (in `main.py`)

*   `DATA_FILE_PATH`: Path to the input CSV file.
*   `COLUMNS_TO_PROCESS`: List of column names in the CSV to process for similarity search.
*   `CHUNK_SIZE`: The number of data points in each time series chunk (default: 50).
*   `OVERLAP`: The number of overlapping data points between consecutive chunks (default: 10).
*   `TOP_K`: The number of similar results to retrieve (default: 3).
*   `CHROMA_DB_PATH`: Directory to store the persistent ChromaDB data (default: `./chroma_db`).
*   `COLLECTION_NAME`: Name of the collection within ChromaDB (default: `timeseries_chunks`).
