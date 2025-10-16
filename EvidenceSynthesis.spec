# EvidenceSynthesis.spec
# Build with: pyinstaller --clean EvidenceSynthesis.spec
block_cipher = None

a = Analysis(
    ['run_app.py'],
    pathex=[],
    binaries=[],
    datas=[('app.py', '.')],  # include your Streamlit app file
    hiddenimports=[
        # Add only if you actually use them at startup; keep most lazy-imported
        # "pdfplumber", "PyPDF2", "fitz", "pdf2image", "pytesseract",
        # "transformers", "sentence_transformers", "matplotlib", "plotly",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='EvidenceSynthesisApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,   # set False if you want no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
