# run_app.py
import sys
import pathlib
import os

def main():
    # Ensure headless defaults for CI/EXE
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")

    app_path = str(pathlib.Path(__file__).with_name("app.py"))
    # Invoke Streamlit programmatically
    from streamlit.web.cli import main as stcli
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli())

if __name__ == "__main__":
    main()
