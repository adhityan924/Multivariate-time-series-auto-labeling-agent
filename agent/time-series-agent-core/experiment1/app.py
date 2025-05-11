import os
import sys
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify, redirect, url_for
from datetime import datetime
import logging

# Add the project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import from the experiment
import config
from src.tools import load_data, setup_vector_store
from src.agent_setup import agent_executor

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the dataset on startup
try:
    df = pd.read_csv(config.DATASET_PATH)
    columns = df.columns.tolist()
    logger.info(f"Dataset loaded successfully with {len(df)} rows and columns: {columns}")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    df = None
    columns = []

@app.route('/')
def index():
    return render_template('index.html', columns=columns)

@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    # Get form inputs
    start_row = int(request.form.get('start_row'))
    end_row = int(request.form.get('end_row'))
    column_name = request.form.get('column_name')
    label = request.form.get('label')
    
    # Log the experiment parameters
    logger.info(f"Running experiment with parameters: Start Row={start_row}, End Row={end_row}, Column={column_name}, Label={label}")
    
    # Prepare input for agent
    input_query = f"Find segments similar to the one from row {start_row} to {end_row} in column '{column_name}', and label them as '{label}'."
    
    agent_input = {
        "input": input_query,
        "input_start_row": start_row,
        "input_end_row": end_row,
        "input_column_name": column_name,
        "input_label": label,
    }
    
    # Create a unique run ID for this experiment
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Execute the agent
    try:
        # Run the agent
        response = agent_executor.invoke(agent_input)
        logger.info(f"Agent execution completed for run {run_id}")
        
        # Process response
        result = {
            'run_id': run_id,
            'parameters': agent_input,
            'raw_response': response,
        }
        
        # Parse output string if available
        if isinstance(response, dict) and 'output' in response:
            final_output_str = response['output']
            
            # Clean the output string if needed
            if final_output_str.startswith("```json"):
                final_output_str = final_output_str[7:]
            if final_output_str.endswith("```"):
                final_output_str = final_output_str[:-3]
            final_output_str = final_output_str.strip()
            
            try:
                parsed_output = json.loads(final_output_str)
                result['parsed_output'] = parsed_output
                
                # Extract key information
                segments = parsed_output.get('identified_segments', [])
                result['segments'] = segments
                result['label'] = parsed_output.get('assigned_label', 'N/A')
                result['explanation'] = parsed_output.get('explanation', 'N/A')
                result['uncertainty'] = parsed_output.get('uncertainty_notes', 'N/A')
                
                # Save results for later visualization
                save_results(run_id, result)
                
                return redirect(url_for('view_results', run_id=run_id))
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse output as JSON: {e}")
                result['error'] = f"Failed to parse output as JSON: {e}"
                return render_template('error.html', error=f"Failed to parse agent output: {e}", output=final_output_str)
        else:
            logger.warning("Agent response missing 'output' key")
            return render_template('error.html', error="Agent response did not contain expected output format", output=str(response))
            
    except Exception as e:
        logger.error(f"Error executing agent: {e}", exc_info=True)
        return render_template('error.html', error=f"Error executing agent: {e}")

def save_results(run_id, results):
    """Save experiment results to disk"""
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, f"{run_id}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved for run {run_id}")

@app.route('/results/<run_id>')
def view_results(run_id):
    """View the results of a specific experiment run"""
    results_file = os.path.join(os.path.dirname(__file__), 'results', f"{run_id}.json")
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        # Generate visualizations
        visualizations = generate_visualizations(results)
        
        return render_template('results.html', 
                              run_id=run_id,
                              parameters=results['parameters'],
                              segments=results.get('segments', []),
                              label=results.get('label', 'N/A'),
                              explanation=results.get('explanation', 'N/A'),
                              uncertainty=results.get('uncertainty', 'N/A'),
                              visualizations=visualizations)
    
    except Exception as e:
        logger.error(f"Error loading results for run {run_id}: {e}", exc_info=True)
        return render_template('error.html', error=f"Error loading results: {e}")

def generate_visualizations(results):
    """Generate visualization data for the results"""
    try:
        # Load the dataset
        df = pd.read_csv(config.DATASET_PATH)
        
        # Extract parameters
        column_name = results['parameters']['input_column_name']
        query_start = results['parameters']['input_start_row']
        query_end = results['parameters']['input_end_row']
        segments = results.get('segments', [])
        
        # Create visualization data
        vis_data = {
            'time_series': create_time_series_visualization(df, column_name, query_start, query_end, segments),
            'agent_steps': extract_agent_steps(results)
        }
        
        return vis_data
    
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}", exc_info=True)
        return {}

def create_time_series_visualization(df, column_name, query_start, query_end, segments):
    """Create time series visualization data"""
    # Create main time series figure
    fig = go.Figure()
    
    # Add the full time series
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df[column_name],
        mode='lines',
        name='Full Time Series',
        line=dict(color='blue', width=1)
    ))
    
    # Highlight the query segment
    fig.add_trace(go.Scatter(
        x=list(range(query_start, query_end + 1)),
        y=df.iloc[query_start:query_end + 1][column_name],
        mode='lines',
        name='Query Segment',
        line=dict(color='red', width=3)
    ))
    
    # Add identified segments
    for i, segment in enumerate(segments):
        start = segment.get('start_row', 0)
        end = segment.get('end_row', 0)
        
        fig.add_trace(go.Scatter(
            x=list(range(start, end + 1)),
            y=df.iloc[start:end + 1][column_name],
            mode='lines',
            name=f'Similar Segment {i+1}',
            line=dict(color='green', width=2)
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Time Series Analysis: {column_name}',
        xaxis_title='Row Index',
        yaxis_title=column_name,
        legend_title='Segments',
        height=600
    )
    
    return fig.to_json()

def extract_agent_steps(results):
    """Extract and format agent steps from the response"""
    steps = []
    
    # Try to extract agent steps from the response
    if 'raw_response' in results and 'intermediate_steps' in results['raw_response']:
        for i, step in enumerate(results['raw_response']['intermediate_steps']):
            if len(step) >= 2:
                action = step[0]
                observation = step[1]
                
                steps.append({
                    'step_number': i + 1,
                    'action': str(action),
                    'observation': str(observation)
                })
    
    return steps

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Run the app on port 5001 to avoid conflicts with AirPlay
    app.run(debug=True, port=5001) 