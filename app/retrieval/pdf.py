"""Getting readable text out of a PDF, and judging whether there is any.

Nothing here touches the database or the network: give it bytes, it gives back
pages, and says plainly when the file cannot be read at all.
"""

from __future__ import annotations

import re
from typing import Sequence


# --- Quality thresholds -----------------------------------------------------
# Deliberately identical to the browser-side check in the portal's textbook
# dialog, so the verdict a teacher saw at upload matches what happens here.
MIN_CHARS_PER_PAGE = 100     # below this the page carries no real text


MIN_ALPHA_RATIO = 0.55       # letters / non-space characters



CHUNK_CHARS = 3500           # ~800-900 tokens


CHUNK_OVERLAP = 500




_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


def clean_page(text: str) -> str:
    """Removes the artefacts PDF extraction reliably produces."""
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)

    text = text.replace("­", "")                       # soft hyphen
    text = re.sub(r"[​-‍﻿]", "", text)       # zero-width
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  # control chars
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)            # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()




def drop_running_headers(pages: list[str]) -> list[str]:
    """
    Strips lines that repeat on most pages - running headers, footers and page
    numbers. Left in, they dominate the chunks and every retrieval returns the
    book's title instead of its content.
    """
    if len(pages) < 5:
        return pages

    counts: dict[str, int] = {}
    for page in pages:
        for line in {l.strip() for l in page.splitlines() if 0 < len(l.strip()) <= 80}:
            counts[line] = counts.get(line, 0) + 1

    threshold = len(pages) * 0.6
    boilerplate = {line for line, n in counts.items() if n >= threshold}

    cleaned = []
    for page in pages:
        kept = [
            l for l in page.splitlines()
            if l.strip() not in boilerplate and not re.fullmatch(r"\s*\d{1,4}\s*", l)
        ]
        cleaned.append("\n".join(kept))
    return cleaned




class PdfUnreadable(Exception):
    """The file itself cannot be opened - corrupt, encrypted, or not a PDF."""




class PdfEncrypted(PdfUnreadable):
    """Password protected. Readable in principle, once the password is removed."""




PAGE_MARKER = re.compile(r"^=+\s*page\s+(\d+)[^\n]*$", re.IGNORECASE | re.MULTILINE)


def looks_like_text(data: bytes) -> bool:
    """
    Is this upload already the words, rather than a document to read them from?

    A PDF always starts "%PDF"; anything that decodes as UTF-8 and does not is
    treated as text. That is what the portal sends for a scanned book it has
    had OCR'd: the pictures are of no use to anyone here, and the text is the
    same book at a fortieth of the size.
    """
    if data[:4] == b"%PDF":
        return False
    # An incremental decoder, because a fixed slice of Malayalam lands in the
    # middle of a three-byte character about two times in three - and a plain
    # decode of that tail raises, which would send real text down the PDF path.
    import codecs

    try:
        codecs.getincrementaldecoder("utf-8")().decode(data[:4096], False)
    except UnicodeDecodeError:
        return False
    return True


def extract_text_pages(data: bytes) -> list[str]:
    """
    Pages out of a recovered-text file.

    Split on the "===== page 137 (ocr, 981 chars) =====" markers the reader
    writes. They are what keeps a citation honest: without them the pages would
    be arbitrary slices and a question said to come from page 137 would be
    pointing at nothing. A file with no markers is one page.
    """
    text = data.decode("utf-8", "replace")
    marks = list(PAGE_MARKER.finditer(text))
    if not marks:
        return [clean_page(text)]

    pages: list[str] = []
    for i, mark in enumerate(marks):
        start = mark.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        pages.append(clean_page(text[start:end]))
    return pages


def extract_pages(pdf_bytes: bytes) -> list[str]:
    """Per-page text, layout-aware so two-column books do not interleave."""
    import fitz  # PyMuPDF

    # Already words: nothing to open, nothing to render.
    if looks_like_text(pdf_bytes):
        return extract_text_pages(pdf_bytes)

    try:
        pages: list[str] = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            # Checked before reading a page. Left to fail on its own it surfaces
            # as "document closed or encrypted" from deep inside the library,
            # which reads as a corrupt file - so the teacher is told to replace
            # a PDF that only needs its password taken off.
            if doc.needs_pass:
                raise PdfEncrypted(
                    "the PDF is password protected. Save an unprotected copy "
                    "and upload that."
                )

            for page in doc:
                pages.append(clean_page(page.get_text("text")))
    except PdfUnreadable:
        raise
    except Exception as error:
        # Raised as its own type so the caller can tell "this file is bad" from
        # "embedding was rate limited", which need opposite answers: one is the
        # uploader's to fix, the other is worth retrying.
        raise PdfUnreadable(str(error)) from error

    return drop_running_headers(pages)




def assess(pages: Sequence[str], from_text: bool = False) -> tuple[str, float]:
    """Returns (status, quality). Mirrors the portal's upload-time check."""
    if not pages:
        return "failed", 0.0

    joined = " ".join(pages)
    # A recovered-text upload has already been through OCR: "needs_ocr" would
    # ask for the one thing that has been done, and the alphabet ratio measures
    # a PDF's broken fonts, which a text file cannot have. Only emptiness is
    # still worth reporting.
    if from_text:
        return ("indexed", 1.0) if len(joined.strip()) else ("failed", 0.0)

    chars_per_page = len(joined) / len(pages)
    non_space = len(re.sub(r"\s", "", joined)) or 1
    letters = len(re.findall(r"[A-Za-zÀ-￿]", joined))
    alpha_ratio = letters / non_space

    if chars_per_page < MIN_CHARS_PER_PAGE:
        return "needs_ocr", alpha_ratio      # scanned: images, no text layer
    if alpha_ratio < MIN_ALPHA_RATIO:
        return "low_quality", alpha_ratio    # broken CID fonts -> gibberish
    return "indexed", alpha_ratio




def chunk_pages(pages: Sequence[str]) -> list[tuple[int, str]]:
    """(page_no, chunk) pairs, overlapped so a sentence is never lost at a seam."""
    out: list[tuple[int, str]] = []
    for page_no, text in enumerate(pages, start=1):
        if not text.strip():
            continue
        start = 0
        while start < len(text):
            piece = text[start:start + CHUNK_CHARS].strip()
            if len(piece) > 80:                    # skip slivers
                out.append((page_no, piece))
            if start + CHUNK_CHARS >= len(text):
                break
            start += CHUNK_CHARS - CHUNK_OVERLAP
    return out
