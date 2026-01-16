
from __future__ import annotations
from pathlib import Path
from typing import Tuple

def extract_text(file_path: str) -> Tuple[str, str]:
    """
    Returns: (text, detected_type)
    Supports: .txt, .md, .docx, .pdf
    """
    p = Path(file_path)
    suf = p.suffix.lower()

    if suf in {".txt", ".md"}:
        return p.read_text(encoding="utf-8", errors="ignore"), suf

    if suf == ".docx":
        from docx import Document
        doc = Document(str(p))
        parts = [para.text for para in doc.paragraphs if para.text and para.text.strip()]
        return "\n\n".join(parts), suf

    if suf == ".pdf":
        import pdfplumber
        texts = []
        with pdfplumber.open(str(p)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    texts.append(t)
        return "\n\n".join(texts), suf

    raise ValueError(f"Unsupported file type: {suf}. Please upload TXT, DOCX, or PDF.")
