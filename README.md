# PRonto Writing Scorecard (MVP)

## What this is
A small Streamlit app that:
1) accepts a TXT/DOCX/PDF upload
2) extracts text
3) calculates writing metrics and section scores
4) generates an overall mark + top improvement areas
5) outputs a downloadable PDF report

This is an MVP intended to be *deployable fast* and then iterated.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (free/cheap options)
- **Streamlit Community Cloud (free)**: push this folder to GitHub, then deploy via Streamlit.
- **Render / Fly.io / Railway**: deploy as a web service running `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Put it on Wix
Wix supports embedding an external web app using an **Embed / iFrame** element.
You can:
- create a landing page (your copy + trust signals + FAQ)
- embed the Streamlit URL for the interactive scorecard.

## Lead capture
MVP stores leads in `leads.csv` on the server. For production, swap this for:
- **Supabase** (free tier) table
- Airtable
- a CRM endpoint (HubSpot forms API, etc.)

## Notes / known limitations
- Passive voice detection is heuristic.
- "Frequency bands" are within-document, not a global English frequency list.
- Scanned PDFs need OCR before upload.
