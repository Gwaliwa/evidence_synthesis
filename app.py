# app.py
import io
import sys
from typing import List, Tuple

import streamlit as st

# Optional deps: we'll import lazily and handle absence gracefully
def _try_import_pdf_engines():
    pdfplumber = None
    pypdf2 = None
    try:
        import pdfplumber as _pdfplumber  # type: ignore
        pdfplumber = _pdfplumber
    except Exception:
        pdfplumber = None
    try:
        import PyPDF2 as _pypdf2  # type: ignore
        pypdf2 = _pypdf2
    except Exception:
        pypdf2 = None
    return pdfplumber, pypdf2


def set_page():
    st.set_page_config(
        page_title="Evidence Synthesis",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("📚 Evidence Synthesis")
    st.caption("Upload PDF(s) to extract text for analysis and synthesis.")


def sidebar():
    st.sidebar.header("⚙️ Settings")
    st.sidebar.write(
        "This lightweight build focuses on PDF text extraction only. "
        "OCR (Tesseract/Poppler) is intentionally excluded to keep the EXE simple."
    )
    return {
        "max_pages": st.sidebar.number_input(
            "Max pages to extract per file (0 = no limit)",
            min_value=0,
            max_value=5000,
            value=0,
            step=1,
        ),
        "show_preview": st.sidebar.checkbox("Show text preview", value=False),
    }


def upload_zone():
    files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Drag & drop PDFs here. They will be processed locally.",
    )
    if not files:
        # Important for CI/EXE smoke-tests: do NOT raise SystemExit; just stop Streamlit safely
        st.info("Upload one or more PDF files to start analysis.")
        st.stop()
    return files


def extract_with_pdfplumber(pdfplumber, file_bytes: bytes, max_pages: int) -> List[Tuple[int, str]]:
    """Return list of (page_number_1_based, text) using pdfplumber."""
    results = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        total = len(pdf.pages)
        cap = total if max_pages in (None, 0) else min(max_pages, total)
        for i in range(cap):
            page = pdf.pages[i]
            txt = page.extract_text() or ""
            results.append((i + 1, txt))
    return results


def extract_with_pypdf2(pypdf2, file_bytes: bytes, max_pages: int) -> List[Tuple[int, str]]:
    """Return list of (page_number_1_based, text) using PyPDF2."""
    results = []
    reader = pypdf2.PdfReader(io.BytesIO(file_bytes))
    total = len(reader.pages)
    cap = total if max_pages in (None, 0) else min(max_pages, total)
    for i in range(cap):
        page = reader.pages[i]
        # extract_text may return None
        txt = page.extract_text() or ""
        results.append((i + 1, txt))
    return results


def extract_pdf_text(file_name: str, file_bytes: bytes, max_pages: int) -> List[Tuple[str, int, str]]:
    """Try pdfplumber first, then PyPDF2. Returns list of (file_name, page, text)."""
    pdfplumber, pypdf2 = _try_import_pdf_engines()

    if pdfplumber is not None:
        try:
            pages = extract_with_pdfplumber(pdfplumber, file_bytes, max_pages)
            return [(file_name, p, t) for (p, t) in pages]
        except Exception as e:
            st.warning(f"pdfplumber failed on {file_name}: {e}")

    if pypdf2 is not None:
        try:
            pages = extract_with_pypdf2(pypdf2, file_bytes, max_pages)
            return [(file_name, p, t) for (p, t) in pages]
        except Exception as e:
            st.warning(f"PyPDF2 failed on {file_name}: {e}")

    # If neither library is available or both failed
    st.error(
        "No PDF extraction backend available. "
        "Install `pdfplumber` or `PyPDF2` in your requirements."
    )
    return []


def main():
    set_page()
    cfg = sidebar()
    files = upload_zone()

    st.success(f"{len(files)} file(s) queued.")
    run = st.button("▶️ Process files", type="primary")
    if not run:
        st.stop()

    import pandas as pd  # local import to keep startup light

    rows: List[Tuple[str, int, str]] = []
    prog = st.progress(0, text="Extracting...")
    for i, f in enumerate(files, start=1):
        fname = f.name
        fbytes = f.read()
        rows.extend(extract_pdf_text(fname, fbytes, cfg["max_pages"]))
        prog.progress(i / len(files), text=f"Extracted: {fname}")

    prog.empty()

    if not rows:
        st.warning("No text extracted. Check that your PDFs contain selectable text (not only scans).")
        st.stop()

    df = pd.DataFrame(rows, columns=["file", "page", "text"])
    st.success("✅ Extraction complete")

    if cfg["show_preview"]:
        st.subheader("Preview")
        st.dataframe(df.head(100), use_container_width=True)

    # CSV download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Download CSV",
        data=csv,
        file_name="evidence_synthesis_text.csv",
        mime="text/csv",
    )

    # Simple per-file summary
    st.subheader("Summary")
    summary = df.groupby("file")["page"].max().reset_index().rename(columns={"page": "pages_extracted"})
    st.dataframe(summary, use_container_width=True)

    with st.expander("ℹ️ Notes"):
        st.markdown(
            "- This build avoids OCR to keep the Windows EXE lightweight.\n"
            "- If your PDFs are scans (images), consider enabling OCR in a future build with Tesseract/Poppler.\n"
            "- For richer analytics (topics, sections, indicators), plug that logic after the extraction step."
        )


if __name__ == "__main__":
    # Streamlit runs the script top-to-bottom; guard is here for clarity when executed directly.
    try:
        main()
    except SystemExit:
        # Avoid killing the PyInstaller process abruptly (rare, but safe).
        pass
