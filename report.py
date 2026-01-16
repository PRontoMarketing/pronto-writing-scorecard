
from __future__ import annotations

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

def build_pdf(result: dict, meta: dict) -> bytes:
    """
    Creates a simple PDF score report.
    result: scorecard result dict
    meta: {"name":..., "email":..., "company":..., "filename":...}
    """
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    def text(x_mm, y_mm, s, size=10):
        c.setFont("Helvetica", size)
        c.drawString(x_mm*mm, y_mm*mm, s)

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(18*mm, (height/mm-20)*mm, "PRonto Writing Scorecard")
    c.setFont("Helvetica", 10)
    c.drawString(18*mm, (height/mm-26)*mm, f"File: {meta.get('filename','')}")
    c.drawString(18*mm, (height/mm-31)*mm, f"Name: {meta.get('name','')}   Email: {meta.get('email','')}   Company: {meta.get('company','')}")

    # Scores
    y = height/mm - 45
    text(18, y, f"Overall score: {result['overall_score']:.1f}/100", size=14); y -= 8
    text(18, y, f"Readability: {result['readability_score']:.1f}"); y -= 6
    text(18, y, f"Sentence structure: {result['sentence_structure_score']:.1f}"); y -= 6
    text(18, y, f"Complexity & style: {result['style_score']:.1f}"); y -= 6
    text(18, y, f"Word usage: {result['word_usage_score']:.1f}"); y -= 10

    # Key stats
    gs = result["general_stats"]
    text(18, y, f"Words: {int(gs['total_words'])}   Sentences: {int(gs['total_sentences'])}   Paragraphs: {int(gs['total_paragraphs'])}   Est. reading time: {gs['reading_time_minutes']:.1f} min"); y -= 10

    # Improvements
    c.setFont("Helvetica-Bold", 12)
    c.drawString(18*mm, y*mm, "Main areas for improvement")
    y -= 6
    c.setFont("Helvetica", 10)
    for tip in result.get("improvements", [])[:6]:
        # wrap
        max_chars = 95
        lines = [tip[i:i+max_chars] for i in range(0, len(tip), max_chars)]
        for ln in lines:
            c.drawString(22*mm, y*mm, f"• {ln}")
            y -= 5
            if y < 20:
                c.showPage()
                y = height/mm - 20
                c.setFont("Helvetica", 10)

    c.showPage()
    c.save()
    return buf.getvalue()
