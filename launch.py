# launch.py — start Streamlit server from a frozen EXE (PyInstaller-safe)
import os
import sys

BASE = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))

def main():
    app_path = os.path.join(BASE, "app.py")
    if not os.path.exists(app_path):
        raise FileNotFoundError(f"Cannot find app.py at {app_path}")

    # Launch Streamlit programmatically (works inside PyInstaller)
    from streamlit.web import cli as stcli

    # Make Streamlit totally headless & predictable for CI smoke test
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
    os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")

    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.headless", "true",
        "--server.address", os.environ["STREAMLIT_SERVER_ADDRESS"],
        "--server.port", os.environ["STREAMLIT_SERVER_PORT"],
        "--browser.gatherUsageStats", "false",
    ]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
