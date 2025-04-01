from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import json
import ast
from pathlib import Path
import pandas as pd
from typing import List
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import TimeSeriesAgent

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="web_app/static"), name="static")
templates = Jinja2Templates(directory="web_app/templates")

# Initialize the agent with default settings
agent = TimeSeriesAgent(
    data_path="./dataset/#1-mini.csv",
    chunk_size=4,
    overlap=2,
    embedding_model="sentence-transformers"
)

# Process the default dataset on startup
try:
    agent.process_data()
    print("Default dataset processed successfully")
except Exception as e:
    print(f"Error processing default dataset: {str(e)}")

@app.get("/")
async def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload."""
    try:
        # Save the uploaded file
        save_path = Path("dataset") / file.filename
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)
        
        # Update agent with new file
        agent.data_path = str(save_path)
        
        # Process the data
        agent.process_data()
        
        return JSONResponse({
            "status": "success",
            "message": f"File {file.filename} uploaded and processed successfully"
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })

@app.post("/analyze")
async def analyze_pattern(
    pattern: str = Form(...),
    label: str = Form(...)
):
    """Analyze a time series pattern."""
    try:
        # Parse the pattern string to a list
        pattern_list = ast.literal_eval(pattern)
        if not isinstance(pattern_list, list):
            raise ValueError("Pattern must be a list of numbers")
        
        # Query the database
        results = agent.query_database({
            "time_series_array": pattern_list,
            "annotation_label": label
        })
        
        return JSONResponse({
            "status": "success",
            "results": results
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 