import os
import sys
import subprocess

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Start Streamlit
streamlit_path = os.path.join(project_root, ".venv", "Scripts", "streamlit.exe")
dashboard_path = os.path.join(project_root, "src", "dashboard_app.py")

subprocess.run([streamlit_path, "run", dashboard_path])