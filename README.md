# Evidence Synthesis Streamlit App (v4.2)

**New in v4.2**
- **Heading → Dimension mapping UI** (boost *or* force) + adjustable boost strength
- **Evidence sentences with page numbers** (auditable trail)
- **Three scoring modes** to avoid “all 10s” saturation:
  - *Curved (anti-saturation)* — default. Uses a saturating curve on hits-per-1k-chars.
  - *Length-normalized* — scales by document length (adjustable pivot).
  - *Classic* — legacy cap at 10 (can saturate quickly).
- **Diagnostics** sheet with extraction method, char counts, and raw hit totals
- Optional PyMuPDF support (if available) but **not required**

## Install (default: no PyMuPDF — macOS 3.13 friendly)
```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
# OCR for scanned PDFs:
# macOS:  brew install tesseract poppler
# Ubuntu:  sudo apt-get install tesseract-ocr poppler-utils
# Windows: install Tesseract & Poppler and set paths in the sidebar
```

## Optional installs
- **macOS + Python 3.13**: `pip install -r requirements-macos-313.txt` (slightly newer pandas/numpy).
- **With PyMuPDF (Python 3.12 recommended)**: `pip install -r requirements-with-fitz.txt`

## Run
```bash
streamlit run app.py
```

## Why did everything become “10” earlier?
v3 used a simple *cap at 10* on raw keyword hits. v4 initially added a length-normalized formula with a small pivot, plus section boosts — which made many scores saturate at 10. v4.2 fixes that by adding a **curved scoring mode** (default) and sliders to tune normalization. If you still see saturation, either lower the boost/τ or switch to the *Length-normalized* mode and increase the pivot.
