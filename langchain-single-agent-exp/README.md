# Time Series Similarity and Annotation Agent

A CLI-based multivariate time series similarity and annotation agent that uses the LangChain framework to orchestrate data loading, chunking, embedding, similarity search, and annotation of time series data.

## Features

- Automatically loads multivariate time series data from CSV files
- Chunks time series data with configurable size and overlap
- Embeds chunks using OpenAI embeddings (with optional open-source alternatives)
- Stores embeddings in a local ChromaDB vector database
- Performs similarity searches on user-provided time series data
- Annotates similar time series patterns

## Installation

### Requirements

- Python 3.12
- Windows 10
- OpenAI API key (for default embeddings)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/time-series-agent.git
   cd time-series-agent
   ```

2. Create a `dataset` folder in the project root and place your time series CSV files there:
   ```
   mkdir dataset
   # Copy your CSV files to the dataset folder
   ```

3. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Set up environment variables:
   - Create a `.env` file in the project root directory
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

## Usage

### Running the CLI

```
python cli.py --data-path "path/to/your/data.csv" --chunk-size 50 --overlap 10
```

### Arguments

- `--data-path`: Path to the CSV file containing time series data (default: "./dataset/#1-mini.csv")
- `--chunk-size`: Size of chunks for time series data (default: 50)
- `--overlap`: Overlap between chunks (default: 10)
- `--top-k`: Number of similar time series chunks to return (default: 3)
- `--embedding-model`: Embedding model to use ("openai" or "sentence-transformers", default: "openai")

### Example

1. Run the CLI:
   ```
   python cli.py
   ```

2. The application will load the data, chunk it, embed it, and store it in the vector database.

3. You'll be prompted to enter a time series array and an annotation label:
   ```
   Enter time series array: [5, 5.1, 5.3, 5.2]
   Enter annotation label: peak-pattern
   ```

4. The application will embed your query, perform a similarity search, and return the top results:
   ```
   {
     "peak-pattern": [
       [5.0, 5.2, 5.3, 5.1],
       [4.9, 5.1, 5.4, 5.2],
       [5.1, 5.3, 5.2, 5.1]
     ]
   }
   ```

## How It Works

1. **Agent-Based Architecture**: The application uses a LangChain agent to orchestrate the entire workflow. The agent is defined in `agent.py` and uses tools for each major operation.

2. **Data Loading**: The agent loads the CSV file specified by the user.

3. **Chunking**: Each column in the CSV (representing a separate time series) is chunked into overlapping windows using the specified chunk size and overlap.

4. **Embedding**: Chunks are converted into vector embeddings using either OpenAI's embedding service or a local sentence-transformers model.

5. **Vector Database**: Embeddings are stored in a ChromaDB collection for efficient similarity search.

6. **User Interaction**: The agent prompts the user for a time series array and an annotation label.

7. **Similarity Search**: The user's input is embedded and used to find similar patterns in the database.

8. **Result Formatting**: The results are formatted as a JSON object with the annotation label as the key and the similar time series chunks as the value.

## Customization

- Place your CSV files in the `dataset` folder or specify a different path using the `--data-path` parameter.
- To use a free/open-source embedding model instead of OpenAI, use the `--embedding-model sentence-transformers` parameter.
- Adjust chunk size and overlap with the `--chunk-size` and `--overlap` parameters to optimize for your specific data.

## License

MIT 