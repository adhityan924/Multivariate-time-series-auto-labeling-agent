# Time Series Similarity and Annotation Agent

A CLI-based agent for finding similar patterns in multivariate time series data and annotating them.

## Features

- Loads time series data from CSV files
- Chunks time series with configurable window size and overlap
- Supports both OpenAI and local embeddings
- Stores embeddings in local ChromaDB vector database
- Interactive CLI for querying similar patterns
- Colorized output for better visibility

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Usage

### Basic Usage
```bash
python cli.py
```

### Using Local Embeddings (instead of OpenAI)
```bash
python cli.py --local
```

### Example Session
```
$ python cli.py
Loading data...
Data loaded successfully!
Processing dataset...
Chunking time series...
Using OpenAI embeddings...

Enter your time series pattern to search for:
Time series array (comma-separated numbers): 5.0,5.1,5.3,5.2
Annotation label: peak-pattern

Similar patterns found:
{
  "peak-pattern": [
    [5.0, 5.2, 5.3, 5.1],
    [4.9, 5.1, 5.4, 5.2],
    [5.1, 5.3, 5.2, 5.1]
  ]
}

Search again? (y/n): n
```

## Configuration

You can modify these parameters in `agents/main_agent.py`:
- `chunk_size`: Size of each time series window (default: 50)
- `overlap`: Overlap between consecutive windows (default: 10)

## Data Format

The CSV file should contain columns of numerical time series data. Each column will be processed independently.

Example CSV structure:
```
timestamp,value1,value2,value3
0,1.2,3.4,5.6
1,1.3,3.5,5.7
...
```

## Dependencies

See `requirements.txt` for complete list. Main dependencies:
- agno-agentic-framework
- chromadb
- openai
- pandas
- numpy
- colorama
