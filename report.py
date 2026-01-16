
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

    def text(x_mm, y_mm, s, size=10, bold=False):
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(x_mm*mm, y_mm*mm, s)

    # Header
    text(18, height/mm-20, "PRonto Writing Scorecard", size=18, bold=True)
    text(18, height/mm-26, f"File: {meta.get('filename','')}", size=10)
    text(18, height/mm-31, f"Name: {meta.get('name','')}   Email: {meta.get('email','')}   Company: {meta.get('company','')}", size=10)

    # Key scores
    y = height/mm - 45
    text(18, y, f"Overall score: {result['overall_score']:.1f}/100", size=14, bold=True); y -= 8

    fre = result["readability"]["flesch_reading_ease"]
    fre_band = result.get("interpretations", {}).get("flesch_reading_ease_band", "")
    text(18, y, f"Flesch Reading Ease: {fre:.1f} ({fre_band})", size=11, bold=True); y -= 6
    text(18, y, "Guide: 90+ very easy • 60–70 standard • 30–50 difficult • <30 very difficult", size=9); y -= 8

    text(18, y, f"Readability score: {result['readability_score']:.1f}/100"); y -= 6
    text(18, y, f"Sentence structure: {result['sentence_structure_score']:.1f}/100"); y -= 6
    text(18, y, f"Style: {result['style_score']:.1f}/100"); y -= 6
    text(18, y, f"Word usage: {result['word_usage_score']:.1f}/100"); y -= 10

    # Stats
    gs = result["general_stats"]
    text(18, y, f"Words: {int(gs['total_words'])}   Sentences: {int(gs['total_sentences'])}   Paragraphs: {int(gs['total_paragraphs'])}   Est. reading time: {gs['reading_time_minutes']:.1f} min"); y -= 10

    # Improvements
    text(18, y, "Main areas for improvement", size=12, bold=True); y -= 6
    c.setFont("Helvetica", 10)
    for tip in result.get("improvements", [])[:6]:
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
