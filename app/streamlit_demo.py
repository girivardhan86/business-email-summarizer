# app/streamlit_demo.py
import sys
from pathlib import Path
import re
import streamlit as st

# allow imports from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.inference import Summarizer

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Email / Chat Summarizer",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("📧 Email / Chat Summarizer & Spam Detector")

# --------------------------------------------------
# Utilities
# --------------------------------------------------
def clean_subject(text: str) -> str:
    """Remove email subject line if present."""
    return re.sub(r"^Subject:.*\n?", "", text, flags=re.IGNORECASE).strip()

# --------------------------------------------------
# Load model (cached)
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading summarization model...")
def load_model():
    return Summarizer()

summarizer = load_model()

# --------------------------------------------------
# UI
# --------------------------------------------------
text = st.text_area(
    "Paste email or chat here:",
    height=350,
    placeholder="Paste a full email or long chat message here..."
)

# --------------------------------------------------
# Action
# --------------------------------------------------
if st.button("🔍 Summarize"):
    cleaned_text = clean_subject(text)

    if not cleaned_text:
        st.warning("Please paste some text first.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarizer.summarize(cleaned_text)

        st.subheader("📝 Summary")
        st.write(summary)

        st.download_button(
            label="⬇️ Download summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    "<hr style='margin-top:2rem'>"
    "<small>Built with 🤗 Transformers & Streamlit</small>",
    unsafe_allow_html=True
)
