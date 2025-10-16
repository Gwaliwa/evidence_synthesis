# app.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Evidence Synthesis", layout="wide")

# --- Session init ---
if "evidence_tables" not in st.session_state:
    st.session_state.evidence_tables = []  # list[pd.DataFrame]
if "evidence_df" not in st.session_state:
    st.session_state.evidence_df = pd.DataFrame()

st.title("📚 Evidence Synthesis")
st.caption("Upload files, then click **Extract Evidence**.")

uploaded_files = st.file_uploader(
    "Upload PDF/DOCX/CSV (multiple allowed)",
    type=["pdf", "docx", "csv"],
    accept_multiple_files=True,
)

# --- Lazy/heavy imports happen in functions only ---
def extract_all(files):
    """Return list[pd.DataFrame] of evidence tables extracted from uploaded files.
       All heavy libraries are imported here to avoid runtime context warnings.
    """
    out = []

    for f in files:
        name = f.name.lower()

        if name.endswith(".csv"):
            import io
            df = pd.read_csv(io.BytesIO(f.getbuffer()))
            df["__source"] = f.name
            out.append(df)

        elif name.endswith(".pdf"):
            # Example stub: replace with your real PDF pipeline
            # Lazy import heavy deps only if you actually use them
            # import pdfplumber
            df = pd.DataFrame({"excerpt": [f"(stub) parsed from {f.name}"], "__source": [f.name]})
            out.append(df)

        elif name.endswith(".docx"):
            # Example stub: replace with docx parsing
            # import docx
            df = pd.DataFrame({"excerpt": [f"(stub) parsed from {f.name}"], "__source": [f.name]})
            out.append(df)

    return out

def build_evidence(tables):
    """Safe concat. Returns empty df if no tables."""
    if not tables:
        return pd.DataFrame()
    try:
        return pd.concat(tables, ignore_index=True)
    except Exception as e:
        st.error(f"Could not combine tables: {e}")
        return pd.DataFrame()

col1, col2 = st.columns([1,1])
with col1:
    run_clicked = st.button("🚀 Extract Evidence", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("🧹 Clear", use_container_width=True)

if clear_clicked:
    st.session_state.evidence_tables = []
    st.session_state.evidence_df = pd.DataFrame()
    st.success("Cleared session state.")

if run_clicked:
    if not uploaded_files:
        st.info("Please upload at least one file, then click **Extract Evidence**.")
        st.stop()
    st.session_state.evidence_tables = extract_all(uploaded_files)
    st.session_state.evidence_df = build_evidence(st.session_state.evidence_tables)

# --- Display section (safe if empty) ---
with st.expander("Evidence (table)", expanded=True):
    if st.session_state.evidence_df.empty:
        st.write("No evidence yet — upload files and click **Extract Evidence**.")
    else:
        st.dataframe(st.session_state.evidence_df, use_container_width=True)

# --- Optional download ---
if not st.session_state.evidence_df.empty:
    st.download_button(
        "⬇️ Download evidence (CSV)",
        data=st.session_state.evidence_df.to_csv(index=False).encode("utf-8"),
        file_name="evidence.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption("Office of Evaluation NYHQ.")
