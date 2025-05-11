# Time Series Auto-Labeling Agent: Advanced UI

This is the advanced version of the Time Series Auto-Labeling Agent UI, offering a comprehensive and feature-rich interface for time series pattern analysis.

## New Features

### 1. Enhanced User Interface
- **Modern Dashboard**: Sleek design with intuitive navigation and responsive layout
- **Dark/Light Mode**: Support for both dark and light themes
- **Interactive Components**: Dynamic elements that respond to user interactions
- **Tabbed Navigation**: Organize information in a clear, accessible way

### 2. Advanced Visualizations
- **Interactive Time Series Plots**: Zoom, pan, and filter data directly in the visualization
- **Segment Heatmap**: Visual overview of segment distribution across the dataset
- **Statistical Comparison**: Radar charts comparing attributes across different segments
- **Batch Results Visualization**: Compare results across multiple experiment runs

### 3. Enhanced Functionality
- **User Authentication**: Secure login system with user-specific dashboards
- **Batch Processing**: Run multiple experiments with different parameters simultaneously
- **Comparison Tool**: Compare results from different experiment runs side by side
- **Export Options**: Export results in multiple formats (JSON, CSV, PDF, images)

### 4. Advanced Analytics
- **Segment Statistics**: In-depth analysis of each identified segment
- **Pattern Similarity Matrix**: Compare similarities between identified segments
- **Uncertainty Analysis**: Quantify and visualize confidence levels
- **Coverage Analysis**: Measure how much of the dataset contains identified patterns

## Installation

1. Ensure you have Python 3.7+ installed
2. Install required dependencies:

```bash
pip install flask pandas plotly numpy
```

3. Set up your OpenAI API key in the `.env` file located in the `milestone4-and5` directory

## Usage

1. Start the application:

```bash
cd milestone4-and5/final-milestone
python app.py
```

2. Open your web browser and navigate to [http://127.0.0.1:5002/](http://127.0.0.1:5002/)

3. Log in with the credentials:
   - Username: `admin`
   - Password: `password123`

4. From the dashboard, you can:
   - Run individual experiments
   - Configure batch processing
   - View previous results
   - Compare experiment outcomes
   - Export results in various formats

## UI Components

### Dashboard
The dashboard provides an overview of:
- Dataset statistics
- Recent experiments
- Batch processing status
- Quick access to commonly used functions

### Experiment Configuration
Configure experiments with:
- Start and end rows for the query segment
- Column selection for analysis
- Label assignment for identified patterns

### Results Visualization
View results with:
- Interactive time series plots
- Segment highlighting
- Statistical comparisons
- Pattern distribution visualization

### Agent Process Transparency
Understand the AI agent's reasoning:
- Step-by-step log of actions and observations
- Decision-making process visibility
- Tool usage details

## Extending the Application

The advanced UI is designed to be extensible:

1. Add new visualization types in `app.py` under the `generate_visualizations()` function
2. Extend the batch processing capabilities in the `process_batch()` function
3. Add new export formats in the export handlers
4. Create additional analytics by adding new functions and corresponding routes

## System Requirements

- Python 3.7+
- Modern web browser with JavaScript enabled
- Internet connection (for OpenAI API)
- Minimum 4GB RAM recommended for larger datasets

## Credits

This advanced UI was developed as part of the Multivariate Time Series Auto-Labeling Agent project, building upon the core functionality established in previous milestones.

## License

MIT License 