# Experiment 2: Plan-and-Solve Agent for Time Series Auto-Labeling

## Overview
This experiment implements a Plan-and-Solve (PS) Agent approach for time series auto-labeling tasks. It extends Experiment 1 by explicitly incorporating a planning phase before execution, while using the same underlying tools and knowledge.

## Implementation Differences from Experiment 1
The key differences compared to Experiment 1 are:

1. **Explicit Planning Phase**: Before executing any actions, the agent first creates a detailed plan that includes:
   - Initial problem analysis
   - Task decomposition
   - Strategy formulation
   - Execution plan
   - Reflection on the plan's effectiveness

2. **Two-Phase Approach**: The agent's work is divided into two distinct phases:
   - Planning Phase: Designing a detailed approach to solve the problem
   - Execution Phase: Following the plan and adjusting as needed

3. **Enhanced Logging**: Comprehensive logging that separately captures both planning and execution phases for better traceability.

4. **Same Tool Set**: Using identical tools as Experiment 1 (data loading, segment retrieval, statistics calculation, and domain knowledge querying) to isolate the impact of the planning phase.

## Expected Improvements
The Plan-and-Solve approach is expected to improve:

1. **Reasoning Quality**: Better decision-making through explicit planning
2. **Traceability**: Clearer record of why actions were taken
3. **Success Rate**: Higher likelihood of correctly labeling time series data
4. **Error Recovery**: Better ability to adjust when initial approaches fail

## Running Instructions

### Prerequisites
- Python 3.12
- Virtual environment with required dependencies (`requirements.txt`)
- OpenAI API key in the environment variable `OPENAI_API_KEY`

### Setup
1. Create and activate the virtual environment:
   ```
   python3.12 -m venv experiment2
   source experiment2/bin/activate  # On Unix/MacOS
   .\experiment2\Scripts\activate   # On Windows
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure environment variables are set:
   ```
   export OPENAI_API_KEY=your_key_here  # On Unix/MacOS
   set OPENAI_API_KEY=your_key_here     # On Windows
   ```

### Running the Experiment
Execute the experiment with:
```
python3.12 run_experiment_2.py
```

Results will be saved in the `logs` directory with timestamps for easy comparison with Experiment 1. 