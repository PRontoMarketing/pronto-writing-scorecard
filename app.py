
import os
import time
import json
import datetime
import pandas as pd
import streamlit as st

from extract import extract_text
from scorecard import score_text

st.set_page_config(page_title="PRonto Writing Scorecard", layout="wide")

st.title("PRonto Writing Scorecard")
st.write("Upload a document, get a clear score, and see the main areas to improve.")

with st.expander("How to read these scores (plain English)", expanded=False):
    st.markdown("""
### Overall score (0–100)
A weighted score designed for **client-facing marketing / B2B writing**:
- **Readability (40%)** – can someone understand it quickly?
- **Sentence structure (25%)** – are sentences clear and not overloaded?
- **Style (20%)** – active voice, not overly “academic”, not too dense
- **Word usage (15%)** – reader focus (“you”), fewer hedging words

### The key metric: Flesch Reading Ease
This is the easiest way to understand “readability” quickly.
- **90+** = very easy (simple, conversational)
- **60–70** = standard (good for most web copy)
- **30–50** = difficult (starts to feel like a report)
- **<30** = very difficult (specialist / academic)

If you only look at one thing first, look at **Flesch**.
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
        "flesch_reading_ease": result["readability"]["flesch_reading_ease"],
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

    # --- top metrics ---
    fre = result["readability"]["flesch_reading_ease"]
    fre_band = result.get("interpretations", {}).get("flesch_reading_ease_band", "")

    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
    c1.metric("Overall", f"{result['overall_score']:.1f}/100")
    c2.metric("Flesch (key)", f"{fre:.1f}")
    c3.metric("Readability", f"{result['readability_score']:.1f}/100")
    c4.metric("Sentence structure", f"{result['sentence_structure_score']:.1f}/100")
    c5.metric("Style & usage", f"{(0.6*result['style_score']+0.4*result['word_usage_score']):.1f}/100")

    st.caption(f"Flesch Reading Ease: **{fre_band}**. Target ~55–75 for most marketing/B2B writing.")

    st.subheader("Main areas for improvement")
    for tip in result["improvements"]:
        st.write(f"• {tip}")

    # --- quick context cards ---
    interp = result.get("interpretations", {})
    st.info(
        f"**What this score means:** {interp.get('sentence_length_simple','')}  \n"
        f"**Very long sentences:** {interp.get('very_long_sentences_simple','')}  \n"
        f"**Passive voice:** {interp.get('passive_voice_simple','')}  \n"
        f"**Reading time:** {interp.get('reading_time_simple','')}"
    )

    left, right = st.columns([1,1])
    with left:
        st.subheader("Readability (what the markers mean)")
        st.markdown(f"""
- **Flesch Reading Ease:** {fre:.1f} (**{fre_band}**)  
- **Flesch–Kincaid Grade:** {result["readability"]["flesch_kincaid_grade"]:.1f} (rough “grade level”)  
- **Gunning Fog:** {result["readability"]["gunning_fog"]:.1f} (rough “education level” needed on first read)  
- **SMOG:** {result["readability"]["smog"]:.1f} (another “education level” estimate)  
- **Coleman–Liau:** {result["readability"]["coleman_liau"]:.1f} (based on letters/sentences)  

**Simple rule:** if **Flesch** is low, you’ll usually see Fog/SMOG climb too.
""")
        with st.expander("Show raw readability numbers"):
            st.json(result["readability"])

        st.subheader("General document stats")
        st.json(result["general_stats"])

    with right:
        st.subheader("Sentence structure (why your score is what it is)")
        st.markdown(f"""
- **Average sentence length:** {result["sentence_structure"]["mean_words_per_sentence"]:.1f} words (target ~14–20)  
- **31+ word sentences:** {int(result["sentence_structure"]["len_31_plus"])} ({result["sentence_structure"]["share_len_31_plus"]*100:.0f}%)  
- **Shortest / longest sentence:** {int(result["sentence_structure"]["min_sentence_length"])} / {int(result["sentence_structure"]["max_sentence_length"])} words  
""")
        with st.expander("Show full sentence length breakdown"):
            st.json(result["sentence_structure"])

        st.subheader("Complexity & style")
        st.json(result["style"])

        st.subheader("Word usage")
        st.json(result["word_usage"])

    # --- downloads ---
    from report import build_pdf
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
