"""Extract text from a PDF.

Usage:
    pip install pypdf
    python pdf_to_text.py path/to/file.pdf
"""

import sys
from pathlib import Path

from pypdf import PdfReader

# Windows consoles default to cp1252 and crash on Unicode; force UTF-8 output.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python pdf_to_text.py <path-to-pdf>")
        raise SystemExit(1)

    pdf_path = sys.argv[1]
    text = extract_text(pdf_path)

    out_path = Path(pdf_path).with_suffix(".txt")
    out_path.write_text(text, encoding="utf-8")

    print(f"Extracted {len(text)} characters -> {out_path}")
    print("\n--- preview ---")
    print(text[:1000])
