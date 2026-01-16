
import os
import time
import json
import datetime
import pandas as pd
import streamlit as st

from extract import extract_text
from scorecard import score_text
from report import build_pdf

st.set_page_config(page_title="PRonto Writing Scorecard", layout="wide")

st.title("PRonto Writing Scorecard (MVP)")
st.write("Upload a document, get a scorecard, and see the main areas to improve.")

with st.expander("What this measures", expanded=False):
    st.markdown("""
- **Readability indices** (Flesch, Flesch–Kincaid, Fog, SMOG, Coleman–Liau)  
- **General document stats** (words, sentences, paragraphs, estimated reading time)  
- **Sentence structure** (sentence length distribution, mean/median/mode)  
- **Complexity & style** (avg letters/syllables per word, heuristic passive voice)  
- **Word usage** (we/you balance, hedging words, subordinators, within-doc frequency bands)
""")

colA, colB = st.columns([1, 1])
with colA:
    name = st.text_input("Name")
    email = st.text_input("Email")
with colB:
    company = st.text_input("Company")
    intended_audience = st.selectbox("Intended audience", ["General public", "Business / B2B", "Technical"], index=1)

uploaded = st.file_uploader("Upload TXT, DOCX, or PDF", type=["txt", "md", "docx", "pdf"])

if uploaded:
    tmp_dir = "tmp_uploads"
    os.makedirs(tmp_dir, exist_ok=True)
    file_path = os.path.join(tmp_dir, f"{int(time.time())}_{uploaded.name}")
    with open(file_path, "wb") as f:
        f.write(uploaded.getbuffer())

    try:
        text, ftype = extract_text(file_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

    if not text.strip():
        st.warning("Couldn't extract any text from that file. If it's a scanned PDF, you'll need OCR first.")
        st.stop()

    result = score_text(text).to_dict()

    # Save lead row locally (MVP). Swap for Supabase/Airtable/CRM in production.
    row = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "name": name,
        "email": email,
        "company": company,
        "filename": uploaded.name,
        "overall_score": result["overall_score"],
        "readability_score": result["readability_score"],
        "sentence_structure_score": result["sentence_structure_score"],
        "style_score": result["style_score"],
        "word_usage_score": result["word_usage_score"],
    }
    leads_path = "leads.csv"
    try:
        if os.path.exists(leads_path):
            df = pd.read_csv(leads_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(leads_path, index=False)
    except Exception:
        pass

    # --- display ---
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
    c1.metric("Overall", f"{result['overall_score']:.1f}/100")
    c2.metric("Readability", f"{result['readability_score']:.1f}")
    c3.metric("Sentence structure", f"{result['sentence_structure_score']:.1f}")
    c4.metric("Style", f"{result['style_score']:.1f}")
    c5.metric("Word usage", f"{result['word_usage_score']:.1f}")

    st.subheader("Main areas for improvement")
    for tip in result["improvements"]:
        st.write(f"• {tip}")

    left, right = st.columns([1,1])
    with left:
        st.subheader("Readability indices")
        st.json(result["readability"])

        st.subheader("General document statistics")
        st.json(result["general_stats"])

    with right:
        st.subheader("Sentence structure analysis")
        st.json(result["sentence_structure"])

        st.subheader("Complexity & style metrics")
        st.json(result["style"])

        st.subheader("Word usage")
        st.json(result["word_usage"])

    # --- downloads ---
    meta = {"name": name, "email": email, "company": company, "filename": uploaded.name}
    pdf_bytes = build_pdf(result, meta)
    st.download_button(
        "Download PDF report",
        data=pdf_bytes,
        file_name=f"pronto_scorecard_{os.path.splitext(uploaded.name)[0]}.pdf",
        mime="application/pdf",
    )

    st.download_button(
        "Download JSON (raw metrics)",
        data=json.dumps(result, indent=2).encode("utf-8"),
        file_name=f"pronto_scorecard_{os.path.splitext(uploaded.name)[0]}.json",
        mime="application/json",
    )
