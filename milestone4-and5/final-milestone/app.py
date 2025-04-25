import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_file
from datetime import datetime
import logging
import uuid
import threading
import io
import zipfile
from functools import wraps

# Add the project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import from the experiment
import milestone4_and5.experiment1.config as base_config
from src.tools import load_data, setup_vector_store
from milestone4_and5.experiment1.src.agent_setup import agent_executor

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# In-memory storage for active batch processes
active_batches = {}

# Authentication (simple version)
# In a real app, use proper auth system with password hashing
USERS = {
    'admin': 'password123',
    'analyst': 'timeseriesdata'
}

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated

# Load the dataset on startup
try:
    df = pd.read_csv(base_config.DATASET_PATH)
    columns = df.columns.tolist()
    logger.info(f"Dataset loaded successfully with {len(df)} rows and columns: {columns}")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    df = None
    columns = []

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in USERS and USERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            flash('You were successfully logged in')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            error = 'Invalid credentials'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You were logged out')
    return redirect(url_for('login'))

@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@requires_auth
def dashboard():
    # Get recent experiments
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    recent_experiments = []
    
    try:
        for file in sorted(os.listdir(results_dir), reverse=True)[:5]:
            if file.endswith('.json'):
                with open(os.path.join(results_dir, file), 'r') as f:
                    result = json.load(f)
                    recent_experiments.append({
                        'id': file.split('.')[0],
                        'date': datetime.strptime(file.split('.')[0], "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S"),
                        'params': result.get('parameters', {}),
                        'segments': len(result.get('segments', []))
                    })
    except Exception as e:
        logger.error(f"Error loading recent experiments: {e}")
    
    # Calculate dataset stats
    stats = {}
    if df is not None:
        stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'null_percent': round(df.isnull().mean().mean() * 100, 2)
        }
    
    # Check status of batch processes
    batch_status = {
        'active': len([b for b in active_batches.values() if b['status'] == 'running']),
        'completed': len([b for b in active_batches.values() if b['status'] == 'completed']),
        'failed': len([b for b in active_batches.values() if b['status'] == 'failed'])
    }
    
    return render_template('dashboard.html', 
                          stats=stats, 
                          recent_experiments=recent_experiments, 
                          batch_status=batch_status,
                          columns=columns)

@app.route('/run_experiment', methods=['POST'])
@requires_auth
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
            'timestamp': datetime.now().isoformat(),
            'user': session.get('username', 'anonymous')
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
                save_results(run_id, result)  # Save even with error
                return render_template('error.html', error=f"Failed to parse agent output: {e}", output=final_output_str)
        else:
            logger.warning("Agent response missing 'output' key")
            result['error'] = "Agent response did not contain expected output format"
            save_results(run_id, result)  # Save even with error
            return render_template('error.html', error="Agent response did not contain expected output format", output=str(response))
            
    except Exception as e:
        logger.error(f"Error executing agent: {e}", exc_info=True)
        # Save the error for history
        save_results(run_id, {
            'run_id': run_id,
            'parameters': agent_input,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'user': session.get('username', 'anonymous')
        })
        return render_template('error.html', error=f"Error executing agent: {e}")

@app.route('/batch', methods=['GET', 'POST'])
@requires_auth
def batch_processing():
    if request.method == 'POST':
        # Create a batch ID
        batch_id = str(uuid.uuid4())
        
        # Parse batch parameters
        start_rows = [int(x) for x in request.form.get('start_rows').split(',')]
        end_rows = [int(x) for x in request.form.get('end_rows').split(',')]
        column_names = request.form.getlist('column_names')
        labels = request.form.getlist('labels')
        
        # Validate parameters
        if len(start_rows) != len(end_rows):
            flash('Number of start rows must match number of end rows')
            return redirect(url_for('batch_processing'))
        
        # Create batch configuration
        batch_config = {
            'id': batch_id,
            'status': 'running',
            'created': datetime.now().isoformat(),
            'user': session.get('username', 'anonymous'),
            'tasks': [],
            'results': {},
            'completed': 0,
            'total': len(start_rows) * len(column_names) * len(labels)
        }
        
        # Configure each experiment
        for start_row, end_row in zip(start_rows, end_rows):
            for column in column_names:
                for label in labels:
                    task_id = f"{start_row}_{end_row}_{column}_{label}"
                    batch_config['tasks'].append({
                        'id': task_id,
                        'start_row': start_row,
                        'end_row': end_row,
                        'column': column,
                        'label': label,
                        'status': 'pending'
                    })
        
        # Store batch config
        active_batches[batch_id] = batch_config
        
        # Start background thread to process batch
        thread = threading.Thread(target=process_batch, args=(batch_id,))
        thread.daemon = True
        thread.start()
        
        flash(f'Batch processing started with ID: {batch_id}')
        return redirect(url_for('batch_status', batch_id=batch_id))
    
    return render_template('batch.html', columns=columns)

def process_batch(batch_id):
    """Background thread to process batch experiments"""
    batch = active_batches[batch_id]
    
    for task in batch['tasks']:
        try:
            # Update task status
            task['status'] = 'running'
            
            # Prepare agent input
            agent_input = {
                "input": f"Find segments similar to the one from row {task['start_row']} to {task['end_row']} in column '{task['column']}', and label them as '{task['label']}'.",
                "input_start_row": task['start_row'],
                "input_end_row": task['end_row'],
                "input_column_name": task['column'],
                "input_label": task['label'],
            }
            
            # Run agent
            response = agent_executor.invoke(agent_input)
            
            # Process response
            if isinstance(response, dict) and 'output' in response:
                final_output_str = response['output']
                
                # Clean output string
                if final_output_str.startswith("```json"):
                    final_output_str = final_output_str[7:]
                if final_output_str.endswith("```"):
                    final_output_str = final_output_str[:-3]
                final_output_str = final_output_str.strip()
                
                # Parse output
                parsed_output = json.loads(final_output_str)
                
                # Store results
                run_id = f"{batch_id}_{task['id']}"
                batch['results'][task['id']] = {
                    'run_id': run_id,
                    'parameters': agent_input,
                    'segments': parsed_output.get('identified_segments', []),
                    'label': parsed_output.get('assigned_label', 'N/A'),
                    'explanation': parsed_output.get('explanation', 'N/A'),
                    'uncertainty': parsed_output.get('uncertainty_notes', 'N/A')
                }
                
                # Update task status
                task['status'] = 'completed'
            else:
                task['status'] = 'failed'
                task['error'] = 'Agent response missing output key'
        
        except Exception as e:
            logger.error(f"Error in batch task {task['id']}: {e}", exc_info=True)
            task['status'] = 'failed'
            task['error'] = str(e)
        
        # Update batch progress
        batch['completed'] += 1
    
    # Update batch status
    if any(task['status'] == 'failed' for task in batch['tasks']):
        batch['status'] = 'completed_with_errors'
    else:
        batch['status'] = 'completed'
    
    # Save batch results
    save_batch_results(batch_id, batch)

@app.route('/batch/<batch_id>')
@requires_auth
def batch_status(batch_id):
    if batch_id not in active_batches:
        flash('Batch ID not found')
        return redirect(url_for('dashboard'))
    
    batch = active_batches[batch_id]
    return render_template('batch_status.html', batch=batch)

@app.route('/batch/<batch_id>/export')
@requires_auth
def export_batch(batch_id):
    if batch_id not in active_batches:
        flash('Batch ID not found')
        return redirect(url_for('dashboard'))
    
    batch = active_batches[batch_id]
    
    # Create in-memory zip file
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        # Add batch metadata
        metadata = {k: v for k, v in batch.items() if k != 'results'}
        zf.writestr('metadata.json', json.dumps(metadata, indent=2))
        
        # Add each result as separate file
        for task_id, result in batch['results'].items():
            zf.writestr(f'results/{task_id}.json', json.dumps(result, indent=2))
    
    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'batch_{batch_id}.zip'
    )

@app.route('/results/<run_id>')
@requires_auth
def view_results(run_id):
    """View the results of a specific experiment run"""
    results_file = os.path.join(os.path.dirname(__file__), 'results', f"{run_id}.json")
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        # Generate visualizations
        visualizations = generate_visualizations(results)
        
        # Get other experiments for comparison
        other_runs = []
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        for file in sorted(os.listdir(results_dir), reverse=True)[:10]:
            if file.endswith('.json') and file != f"{run_id}.json":
                other_runs.append(file.split('.')[0])
        
        return render_template('results.html', 
                              run_id=run_id,
                              parameters=results['parameters'],
                              segments=results.get('segments', []),
                              label=results.get('label', 'N/A'),
                              explanation=results.get('explanation', 'N/A'),
                              uncertainty=results.get('uncertainty', 'N/A'),
                              visualizations=visualizations,
                              other_runs=other_runs,
                              timestamp=results.get('timestamp', 'Unknown'),
                              user=results.get('user', 'anonymous'))
    
    except Exception as e:
        logger.error(f"Error loading results for run {run_id}: {e}", exc_info=True)
        return render_template('error.html', error=f"Error loading results: {e}")

@app.route('/compare')
@requires_auth
def compare_results():
    """Compare multiple experiment results"""
    run_ids = request.args.getlist('runs')
    if not run_ids:
        flash('No runs selected for comparison')
        return redirect(url_for('dashboard'))
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    all_results = []
    
    for run_id in run_ids:
        try:
            with open(os.path.join(results_dir, f"{run_id}.json"), 'r') as f:
                results = json.load(f)
                all_results.append(results)
        except Exception as e:
            logger.error(f"Error loading results for run {run_id}: {e}")
    
    if not all_results:
        flash('No valid results found for comparison')
        return redirect(url_for('dashboard'))
    
    # Generate comparison visualizations
    comparison_viz = generate_comparison_visualizations(all_results)
    
    return render_template('comparison.html',
                         run_ids=run_ids,
                         results=all_results,
                         visualizations=comparison_viz)

def save_results(run_id, results):
    """Save experiment results to disk"""
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, f"{run_id}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved for run {run_id}")

def save_batch_results(batch_id, batch):
    """Save batch results to disk"""
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'batches')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, f"{batch_id}.json"), 'w') as f:
        json.dump(batch, f, indent=2)
    
    logger.info(f"Batch results saved for batch {batch_id}")

def generate_visualizations(results):
    """Generate visualization data for the results"""
    try:
        # Load the dataset
        df = pd.read_csv(base_config.DATASET_PATH)
        
        # Extract parameters
        column_name = results['parameters']['input_column_name']
        query_start = results['parameters']['input_start_row']
        query_end = results['parameters']['input_end_row']
        segments = results.get('segments', [])
        
        # Create visualization data
        vis_data = {
            'time_series': create_time_series_visualization(df, column_name, query_start, query_end, segments),
            'segment_heatmap': create_segment_heatmap(df, segments),
            'statistics': create_statistics_visualization(df, column_name, query_start, query_end, segments),
            'agent_steps': extract_agent_steps(results)
        }
        
        return vis_data
    
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}", exc_info=True)
        return {}

def create_time_series_visualization(df, column_name, query_start, query_end, segments):
    """Create enhanced time series visualization data"""
    # Create main time series figure
    fig = go.Figure()
    
    # Add the full time series
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df[column_name],
        mode='lines',
        name='Full Time Series',
        line=dict(color='rgba(0, 0, 255, 0.3)', width=1)
    ))
    
    # Highlight the query segment
    fig.add_trace(go.Scatter(
        x=list(range(query_start, query_end + 1)),
        y=df.iloc[query_start:query_end + 1][column_name],
        mode='lines',
        name='Query Segment',
        line=dict(color='red', width=3)
    ))
    
    # Add identified segments with different colors
    colors = ['green', 'orange', 'purple', 'brown', 'pink', 'cyan']
    for i, segment in enumerate(segments):
        start = segment.get('start_row', 0)
        end = segment.get('end_row', 0)
        
        color_idx = i % len(colors)
        
        fig.add_trace(go.Scatter(
            x=list(range(start, end + 1)),
            y=df.iloc[start:end + 1][column_name],
            mode='lines',
            name=f'Similar Segment {i+1}',
            line=dict(color=colors[color_idx], width=2)
        ))
    
    # Add segment markers for easier identification
    for i, segment in enumerate(segments):
        start = segment.get('start_row', 0)
        end = segment.get('end_row', 0)
        
        # Add a marker for start and end of segment
        fig.add_trace(go.Scatter(
            x=[start, end],
            y=[df.iloc[start][column_name], df.iloc[end][column_name]],
            mode='markers',
            marker=dict(size=8, symbol='circle', color='black'),
            showlegend=False
        ))
    
    # Add annotations for segments
    for i, segment in enumerate(segments):
        start = segment.get('start_row', 0)
        end = segment.get('end_row', 0)
        mid_point = (start + end) // 2
        
        fig.add_annotation(
            x=mid_point,
            y=df.iloc[mid_point][column_name],
            text=f"S{i+1}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
    
    # Update layout with more detailed settings
    fig.update_layout(
        title=f'Time Series Analysis: {column_name}',
        xaxis_title='Row Index',
        yaxis_title=column_name,
        legend_title='Segments',
        height=600,
        hovermode='closest',
        template='plotly_white',
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='linear'
        )
    )
    
    return fig.to_json()

def create_segment_heatmap(df, segments):
    """Create a heatmap visualization of segments"""
    if not segments:
        return None
    
    # Create a mask array (0 = not in segment, segment_id = in segment)
    mask = np.zeros(len(df))
    
    for i, segment in enumerate(segments):
        start = segment.get('start_row', 0)
        end = segment.get('end_row', 0)
        mask[start:end+1] = i + 1
    
    # Create a figure
    fig = go.Figure()
    
    # Add a heatmap
    fig.add_trace(go.Heatmap(
        z=[mask],
        x=list(range(len(df))),
        y=['Segments'],
        colorscale=[
            [0, 'white'],  # No segment
            [0.1, 'green'],  # Segment 1
            [0.3, 'yellow'],  # Segment 2
            [0.5, 'orange'],  # Segment 3
            [0.7, 'red'],   # Segment 4
            [0.9, 'purple'] # Segment 5+
        ],
        showscale=False,
        hoverinfo='text',
        text=[f'Row {i}: {"Segment " + str(int(v)) if v > 0 else "No segment"}' for i, v in enumerate(mask)]
    ))
    
    # Update layout
    fig.update_layout(
        title='Segment Distribution Overview',
        xaxis_title='Row Index',
        height=150,
        margin=dict(l=50, r=50, t=50, b=30),
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='linear'
        )
    )
    
    return fig.to_json()

def create_statistics_visualization(df, column_name, query_start, query_end, segments):
    """Create visualization of statistics comparison between segments"""
    # Calculate statistics for query segment
    query_data = df.iloc[query_start:query_end+1][column_name]
    query_stats = {
        'mean': query_data.mean(),
        'std': query_data.std(),
        'min': query_data.min(),
        'max': query_data.max(),
        'range': query_data.max() - query_data.min(),
        'median': query_data.median()
    }
    
    # Calculate statistics for each segment
    segment_stats = []
    for i, segment in enumerate(segments):
        start = segment.get('start_row', 0)
        end = segment.get('end_row', 0)
        segment_data = df.iloc[start:end+1][column_name]
        
        segment_stats.append({
            'id': i + 1,
            'start': start,
            'end': end,
            'mean': segment_data.mean(),
            'std': segment_data.std(),
            'min': segment_data.min(),
            'max': segment_data.max(),
            'range': segment_data.max() - segment_data.min(),
            'median': segment_data.median()
        })
    
    # Create radar chart for comparison
    fig = go.Figure()
    
    # Add query segment
    fig.add_trace(go.Scatterpolar(
        r=[query_stats['mean'], query_stats['std'], query_stats['range'], 
           query_stats['max'], query_stats['min'], query_stats['median']],
        theta=['Mean', 'Std Dev', 'Range', 'Max', 'Min', 'Median'],
        fill='toself',
        name='Query Segment'
    ))
    
    # Add identified segments
    for i, stats in enumerate(segment_stats):
        fig.add_trace(go.Scatterpolar(
            r=[stats['mean'], stats['std'], stats['range'], 
               stats['max'], stats['min'], stats['median']],
            theta=['Mean', 'Std Dev', 'Range', 'Max', 'Min', 'Median'],
            fill='toself',
            name=f'Segment {stats["id"]}'
        ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max([
                    query_stats['max'], 
                    query_stats['mean'] + 2*query_stats['std'],
                    *[s['max'] for s in segment_stats],
                    *[s['mean'] + 2*s['std'] for s in segment_stats]
                ])]
            )),
        title='Statistical Comparison of Segments',
        height=500,
        showlegend=True
    )
    
    return fig.to_json()

def generate_comparison_visualizations(results_list):
    """Generate visualizations for comparing multiple experiment results"""
    try:
        # Load the dataset
        df = pd.read_csv(base_config.DATASET_PATH)
        
        # Extract all segments from all results
        all_segments = []
        for i, result in enumerate(results_list):
            segments = result.get('segments', [])
            for segment in segments:
                all_segments.append({
                    'experiment': i,
                    'start_row': segment.get('start_row', 0),
                    'end_row': segment.get('end_row', 0),
                    'label': result.get('label', 'N/A')
                })
        
        # Create comparison visualizations
        vis_data = {
            'overlap_chart': create_overlap_visualization(df, results_list),
            'segment_distribution': create_segment_distribution(all_segments),
            'segment_lengths': create_segment_length_comparison(all_segments)
        }
        
        return vis_data
    
    except Exception as e:
        logger.error(f"Error generating comparison visualizations: {e}", exc_info=True)
        return {}

def create_overlap_visualization(df, results_list):
    """Create visualization showing overlap between experiments"""
    # First, create a visualization that shows all experiments on the same timeline
    fig = go.Figure()
    
    # Add the full time series as background
    if 'parameters' in results_list[0] and 'input_column_name' in results_list[0]['parameters']:
        default_column = results_list[0]['parameters']['input_column_name']
    else:
        default_column = df.columns[0]
    
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df[default_column],
        mode='lines',
        name='Full Time Series',
        line=dict(color='rgba(200, 200, 200, 0.3)', width=1)
    ))
    
    # Add each experiment's segments with different colors
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    for i, result in enumerate(results_list):
        if 'parameters' not in result or 'segments' not in result:
            continue
            
        params = result['parameters']
        segments = result['segments']
        column_name = params.get('input_column_name', default_column)
        color = colors[i % len(colors)]
        
        # Add query segment
        query_start = params.get('input_start_row', 0)
        query_end = params.get('input_end_row', 0)
        
        fig.add_trace(go.Scatter(
            x=list(range(query_start, query_end + 1)),
            y=df.iloc[query_start:query_end + 1][column_name],
            mode='lines',
            name=f'Query {i+1}',
            line=dict(color=color, width=2, dash='dash')
        ))
        
        # Add identified segments
        for j, segment in enumerate(segments):
            start = segment.get('start_row', 0)
            end = segment.get('end_row', 0)
            
            fig.add_trace(go.Scatter(
                x=list(range(start, end + 1)),
                y=df.iloc[start:end + 1][column_name],
                mode='lines',
                name=f'Exp {i+1} Segment {j+1}',
                line=dict(color=color, width=1.5)
            ))
    
    # Update layout
    fig.update_layout(
        title='Comparison of Experiment Results',
        xaxis_title='Row Index',
        yaxis_title='Value',
        height=600,
        legend_title='Experiments',
        template='plotly_white'
    )
    
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='linear'
        )
    )
    
    return fig.to_json()

def create_segment_distribution(all_segments):
    """Create visualization showing the distribution of segments across the dataset"""
    if not all_segments:
        return None
    
    # Get experiment count
    exp_count = max([s['experiment'] for s in all_segments]) + 1
    
    # Prepare data for chart
    labels = [f'Exp {i+1}' for i in range(exp_count)]
    segments_per_exp = [len([s for s in all_segments if s['experiment'] == i]) for i in range(exp_count)]
    
    # Create figure
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=segments_per_exp,
            text=segments_per_exp,
            textposition='auto',
            marker_color='skyblue'
        )
    ])
    
    # Update layout
    fig.update_layout(
        title='Number of Segments Identified per Experiment',
        xaxis_title='Experiment',
        yaxis_title='Number of Segments',
        height=400,
        template='plotly_white'
    )
    
    return fig.to_json()

def create_segment_length_comparison(all_segments):
    """Create visualization comparing segment lengths across experiments"""
    if not all_segments:
        return None
    
    # Calculate segment lengths
    for segment in all_segments:
        segment['length'] = segment['end_row'] - segment['start_row'] + 1
    
    # Create figure
    fig = go.Figure()
    
    # Add box plot per experiment
    exp_count = max([s['experiment'] for s in all_segments]) + 1
    
    for i in range(exp_count):
        exp_segments = [s for s in all_segments if s['experiment'] == i]
        if exp_segments:
            lengths = [s['length'] for s in exp_segments]
            
            fig.add_trace(go.Box(
                y=lengths,
                name=f'Exp {i+1}',
                boxmean=True,
                jitter=0.3,
                pointpos=-1.8,
                boxpoints='all'
            ))
    
    # Update layout
    fig.update_layout(
        title='Segment Length Distribution by Experiment',
        xaxis_title='Experiment',
        yaxis_title='Segment Length (rows)',
        height=400,
        template='plotly_white'
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
    # Create necessary directories
    for directory in ['templates', 'static/css', 'static/js', 'static/img', 'results', 'results/batches']:
        dir_path = os.path.join(os.path.dirname(__file__), directory)
        os.makedirs(dir_path, exist_ok=True)
    
    # Run the app on port 5002 to avoid conflicts
    app.run(debug=True, port=5002) 