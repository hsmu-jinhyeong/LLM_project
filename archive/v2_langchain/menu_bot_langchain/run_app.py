"""Launch script for Menu Recommendation Streamlit app.

Usage:
    python run_app.py
    
    Or directly:
    streamlit run app.py
"""
import subprocess
import sys
from pathlib import Path

def main():
    app_path = Path(__file__).parent / "app.py"
    
    if not app_path.exists():
        print(f"❌ Error: {app_path} not found!")
        sys.exit(1)
    
    print("🚀 Launching Menu Recommendation App...")
    print(f"📁 App: {app_path}")
    print("─" * 60)
    
    try:
        subprocess.run([
            "streamlit", "run", str(app_path),
            "--server.port", "8502",
            "--server.headless", "true"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: Streamlit not found!")
        print("Install it with: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
