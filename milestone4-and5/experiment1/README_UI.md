# Time Series Auto-Labeling Agent UI

This UI provides a web interface for running time series pattern identification experiments using an AI agent powered by LangChain and OpenAI.

## Features

- **Simple Configuration**: Easily set parameters for the experiment through a web form
- **Interactive Visualizations**: View time series data with identified patterns highlighted
- **Agent Process Transparency**: See the step-by-step reasoning of the AI agent
- **Detailed Results**: Get comprehensive information about identified segments and explanations

## Installation

1. Ensure you have Python 3.7+ installed
2. Set up the required dependencies by running:

```bash
python setup_ui.py
```

This will install the necessary packages (Flask, Pandas, Plotly) and create the required directories.

## Usage

1. Start the UI server:

```bash
python app.py
```

2. Open your web browser and navigate to: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

3. Configure your experiment:
   - **Start Row & End Row**: Define the segment to be used as the query pattern
   - **Column Name**: Select the time series column to analyze
   - **Label**: Set the label to assign to similar patterns

4. Click "Run Experiment" to start the analysis

5. View the results:
   - The time series visualization will show the query segment and any identified similar segments
   - The explanation section provides the agent's reasoning
   - The agent steps section shows the detailed process the agent followed

## System Requirements

- Python 3.7+
- Internet connection (for OpenAI API access)
- Valid OpenAI API key set in the `.env` file

## File Structure

- `app.py`: The main Flask application
- `templates/`: HTML templates for the UI
- `results/`: Directory where experiment results are stored
- `setup_ui.py`: Helper script for setting up dependencies

## Troubleshooting

If you encounter issues:

1. Ensure your OpenAI API key is correctly set in the `.env` file
2. Check that the dataset is properly formatted and accessible
3. Verify that all dependencies are installed
4. Look at the console output for any error messages

For detailed error information, check the browser's console and the server logs.

## Extending the UI

To extend the UI with additional features:

1. Add new routes or modify existing ones in `app.py`
2. Update the templates in the `templates/` directory
3. Add new visualization types to the `generate_visualizations()` function

## Notes

- Experiment results are saved in the `results/` directory with a timestamp-based ID
- You can review past experiments by accessing their URLs directly
- The agent uses OpenAI's API, so each experiment will consume API credits 