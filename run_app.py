# run_app.py
import os
import sys

# Respect CI/env defaults but set safe fallbacks
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

# Build the argv just like "streamlit run app.py --server.port=8501 …"
app_path = os.path.join(os.path.dirname(__file__), "app.py")
sys.argv = [
    "streamlit", "run", app_path,
    f"--server.port={os.environ['STREAMLIT_SERVER_PORT']}",
    f"--server.address={os.environ['STREAMLIT_SERVER_ADDRESS']}",
    f"--server.headless={os.environ['STREAMLIT_SERVER_HEADLESS']}",
    f"--browser.gatherUsageStats={os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS']}",
]

from streamlit.web import cli as stcli
if __name__ == "__main__":
    sys.exit(stcli.main())
