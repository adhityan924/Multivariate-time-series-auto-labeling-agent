import os
import sys
import subprocess
import importlib.util

def check_dependency(package_name):
    """Check if a Python package is installed."""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"Package {package_name} is not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Successfully installed {package_name}")
            return True
        except Exception as e:
            print(f"Failed to install {package_name}: {e}")
            return False
    else:
        print(f"Package {package_name} is already installed.")
        return True

def setup_directories():
    """Setup necessary directories for the UI application."""
    dirs = [
        "templates",
        "results"
    ]
    
    for directory in dirs:
        path = os.path.join(os.path.dirname(__file__), directory)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")

def main():
    """Main setup function."""
    print("Setting up Time Series Auto-Labeling UI...")
    
    # Check dependencies
    dependencies = ["flask", "pandas", "plotly"]
    all_installed = True
    
    for dep in dependencies:
        if not check_dependency(dep):
            all_installed = False
    
    if not all_installed:
        print("Warning: Not all dependencies could be installed. Some features may not work correctly.")
    
    # Setup directories
    setup_directories()
    
    print("\nSetup complete! You can start the UI by running:")
    print("python app.py")
    print("\nThen open your browser and go to: http://127.0.0.1:5000/")

if __name__ == "__main__":
    main() 