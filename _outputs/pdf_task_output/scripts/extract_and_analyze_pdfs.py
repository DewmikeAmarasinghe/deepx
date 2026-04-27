#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import io

OUT_DIR = Path.cwd() / '_outputs' / 'pdf_task_output'
TABLES_DIR = OUT_DIR / 'tables'
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

PDFS = [
    ('attention', 'test_demo/pdfs/attention.pdf'),
    ('gpt4', 'test_demo/pdfs/gpt4.pdf')
]

use_ocr = False

# Try to import pdfplumber, pandas, pypdf, pytesseract/pdf2image if available
try:
    import pdfplumber
except Exception as e:
    print('pdfplumber not available:', e, file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
except Exception as e:
    print('pandas not available:', e, file=sys.stderr)
    sys.exit(1)

try:
    from pypdf import PdfReader, PdfWriter
except Exception as e:
    # Try pypdf from PyPDF2
    try:
        from PyPDF2 import PdfReader, PdfWriter
    except Exception as e2:
        print('pypdf/PyPDF2 not available:', e2, file=sys.stderr)
        sys.exit(1)

# OCR libraries
have_ocr = True
try:
    import pytesseract
    from pdf2image import convert_from_path
except Exception:
    have_ocr = False

for short, path in PDFS:
    txt_out = OUT_DIR / f'{short}.txt'
    tables_prefix = TABLES_DIR / f'{short}_table'

    text_parts = []
    table_count = 0
    ocr_pages = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # Extract text
            page_text = page.extract_text()
            if not page_text or len(page_text.strip()) == 0:
                # Try OCR if available
                if have_ocr:
                    try:
                        images = convert_from_path(path, first_page=i, last_page=i)
                        if images:
                            ocr_text = pytesseract.image_to_string(images[0])
                            page_text = ocr_text
                            ocr_pages.append(i)
                    except Exception as e:
                        # OCR failed for this page
                        page_text = ''
                else:
                    page_text = ''

            text_parts.append(f'----PAGE {i}----\n')
            text_parts.append(page_text or '')
            text_parts.append('\n\n')

            # Extract tables
            try:
                tables = page.extract_tables()
            except Exception as e:
                tables = []
            for t in (tables or []):
                # Convert to DataFrame if possible
                table_count += 1
                # t is list of rows (list of lists)
                # Some rows may be None; coerce to empty strings
                rows = [[(cell if cell is not None else '') for cell in row] for row in t]
                # Choose header if first row seems header (heuristic: contains non-numeric)
                df = None
                if len(rows) >= 2:
                    df = pd.DataFrame(rows[1:], columns=rows[0])
                else:
                    df = pd.DataFrame(rows)
                csv_path = TABLES_DIR / f'{short}_table{table_count}.csv'
                # Save without index, preserve formatting
                df.to_csv(csv_path, index=False)

    with open(txt_out, 'w', encoding='utf-8') as f:
        f.write(''.join(text_parts))

    # record ocr info
    if ocr_pages:
        with open(OUT_DIR / f'{short}_ocr_note.txt', 'w', encoding='utf-8') as f:
            f.write(f'OCR performed on pages: {ocr_pages}\n')

# Merge PDFs into one normalized PDF
merged_path = OUT_DIR / 'merged_documents.pdf'
writer = PdfWriter()
for short, path in PDFS:
    try:
        reader = PdfReader(path)
        for p in reader.pages:
            writer.add_page(p)
    except Exception as e:
        print('Failed to read', path, e, file=sys.stderr)
with open(merged_path, 'wb') as f:
    writer.write(f)

print('Done')
