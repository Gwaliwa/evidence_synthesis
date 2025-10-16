# launch.py — start Streamlit server from a frozen EXE
import os, sys, subprocess, shutil, tempfile

# When frozen (PyInstaller onefile), sources are in a temp dir exposed via _MEIPASS
BASE = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))

def main():
    # Ensure we have a real filesystem path to app.py for Streamlit to run.
    src = os.path.join(BASE, "app.py")
    if not os.path.exists(src):
        # If bundled as data with a different name/path, adjust here.
        raise FileNotFoundError(f"Cannot find app.py at {src}")

    # Spawn "python -m streamlit run app.py ..." using the embedded interpreter
    py = sys.executable  # works in frozen EXE
    args = [
        py, "-m", "streamlit", "run", src,
        "--server.headless", "true",
        "--server.address", "127.0.0.1",
        "--server.port", os.environ.get("PORT","8501"),
        "--browser.gatherUsageStats", "false",
    ]
    # In onefile EXE, let it inherit stdio (no console if built with --noconsole)
    os.execv(py, args)  # replace current process

if __name__ == "__main__":
    main()
