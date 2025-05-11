# Time Series Auto-Labeling Agent

An advanced system for automated pattern detection and labeling in multivariate time series data.

## Project Structure

The project is organized into two main components:

### 1. Time Series Web App (`time-series-web-app/`)

A modern Flask web application that provides a user-friendly interface for:
- Uploading and managing time series datasets (supports CSV, Excel, JSON)
- Running pattern detection experiments
- Visualizing results through interactive charts
- Batch processing multiple experiments
- Comparing experiment results

### 2. Agent Core (`agent/time-series-agent-core/`)

The core time series analysis engine that:
- Processes multivariate time series data
- Identifies patterns and segments based on user queries
- Applies labels to similar patterns
- Provides explanations for detected patterns

## Getting Started

### Prerequisites
- Python 3.7+
- Virtual Environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/adhityan924/Multivariate-time-series-auto-labeling-agent.git
cd Multivariate-time-series-auto-labeling-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required packages:
```bash
cd time-series-web-app
pip install -r requirements.txt
```

4. Set up your environment variables by creating a `.env` file:
```
FLASK_SECRET_KEY=your_secret_key
DATASET_PATH=data/sample_timeseries.csv
OPENAI_API_KEY=your_openai_api_key
UPLOAD_FOLDER=data/uploads

# Optional S3 configuration
USE_S3_STORAGE=false
S3_BUCKET_NAME=your-bucket-name
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
```

5. Run the web application:
```bash
cd time-series-web-app
python app.py
```

6. Open your browser and navigate to [http://127.0.0.1:5002/](http://127.0.0.1:5002/)

## Features

- **Dataset Management**: Upload and manage multiple time series datasets
- **Interactive Visualizations**: Dynamic time series plots and statistical analysis
- **Pattern Detection**: Identify similar patterns in time series data
- **Auto-Labeling**: Automatically label similar segments
- **Batch Processing**: Run multiple experiments with different parameters
- **Result Comparison**: Compare multiple experiments side by side

## License

This project is licensed under the MIT License - see the LICENSE file for details.