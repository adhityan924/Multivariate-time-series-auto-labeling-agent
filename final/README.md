# Time Series Auto-Labeling Agent

This is an advanced web application for time series pattern analysis using AI. The application allows users to identify and label similar patterns in time series data.

## Features

- **Modern UI**: Sleek design with responsive layout and dark/light mode
- **Interactive Visualizations**: Dynamic time series plots, segment heatmaps, and statistical comparisons
- **Dataset Management**: Upload and manage multiple time series datasets with support for CSV, Excel, and JSON formats
- **Batch Processing**: Run multiple experiments with different parameters
- **Result Comparison**: Compare multiple experiment results side by side
- **Authentication**: Secure user login system

## Installation

### Prerequisites

- Python 3.7+
- Virtual Environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/time-series-auto-labeling-agent.git
cd time-series-auto-labeling-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required packages:
```bash
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
```

## Usage

1. Start the application:
```bash
cd final
python app.py
```

2. Open your web browser and go to [http://127.0.0.1:5002/](http://127.0.0.1:5002/)

3. Log in with the default credentials:
   - Username: `admin`
   - Password: `password123`

## Dataset Management

The application supports two ways to handle datasets:

### Local File Storage (Default)

By default, uploaded files are stored in the `data/uploads` directory on your local server. This is suitable for development and testing.

### Amazon S3 Storage (Optional)

For production environments or when dealing with large datasets, you can enable Amazon S3 storage:

1. Install the boto3 package: `pip install boto3`
2. Configure AWS credentials (using AWS CLI or environment variables)
3. Update your `.env` file:
```
USE_S3_STORAGE=true
S3_BUCKET_NAME=your-bucket-name
```

### Supported File Formats

- CSV (.csv)
- Excel (.xlsx, .xls)
- JSON (.json)
- Text (.txt)

Maximum file size: 50MB

## Running Experiments

1. From the dashboard, click on "New Experiment"
2. Enter the parameters:
   - Start Row: The beginning row of your query segment
   - End Row: The ending row of your query segment
   - Column Name: Select the column to analyze
   - Label: Enter a label for the identified patterns
3. Click "Run Experiment"
4. View the results with interactive visualizations

## Batch Processing

1. Go to the Batch Processing page
2. Enter multiple start/end row pairs, columns, and labels
3. The system will run all combinations of these parameters
4. Results can be viewed and compared

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 