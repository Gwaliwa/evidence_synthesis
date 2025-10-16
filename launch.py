# launch.py — start Streamlit server from a frozen EXE (PyInstaller-safe)
import os
import sys

# Where the unpacked files live when frozen
BASE = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))

def main():
    app_path = os.path.join(BASE, "app.py")
    if not os.path.exists(app_path):
        raise FileNotFoundError(f"Cannot find app.py at {app_path}")

    # Programmatic entry into Streamlit CLI (works in frozen apps)
    from streamlit.web import cli as stcli

    # Build argv just like "streamlit run app.py ..."
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.headless", "true",
        "--server.address", "127.0.0.1",
        "--server.port", os.environ.get("PORT", "8501"),
        "--browser.gatherUsageStats", "false",
    ]
    # This starts the Streamlit server and blocks
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
