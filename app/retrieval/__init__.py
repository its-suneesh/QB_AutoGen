"""Retrieval of real textbook passages for question generation.

Why this exists
---------------
The prompt used to receive only book TITLES, so the model wrote questions from
whatever it already knew about a book with that name. The PDFs the college
actually prescribes were uploaded, stored and never read. This package opens
them, indexes them once, and returns the passages that match the unit being
examined, so generated questions come from the prescribed text.

Design notes
------------
* Key is the file stem, e.g. "1_734" -> paper 734, book slot 1. That name is
  unique at any moment but NOT permanent: the portal names files by row
  position, so a delete or reorder can hand "1_734.pdf" to a different book.
  The content hash detects exactly that and re-indexes, which is why nothing in
  the portal has to change.

* Everything fails OPEN. If Postgres is down, a PDF 404s or extraction is poor,
  the caller gets no passages and question generation proceeds exactly as it
  does today. Retrieval must never take the generator offline.

The modules split by what they touch: pdf (bytes only), embeddings (the model
API), store (the database), search (reading), documents (writing).
"""

from .documents import (
    STATUS_MESSAGES,
    books_from_payload,
    document_status,
    fetch_pdf,
    index_pdf_bytes,
    prune_documents,
    retrieve_for_generation,
    sync_book,
)
from .embeddings import EMBED_DIM, embed_texts
from .models import BookRef, Passage, parse_doc_id
from .pdf import (
    PdfEncrypted,
    PdfUnreadable,
    assess,
    chunk_pages,
    clean_page,
    drop_running_headers,
    extract_pages,
)
from .passages import format_passages, search, split_topics
from .store import SCHEMA_VERSION, ensure_schema

__all__ = [
    "STATUS_MESSAGES", "SCHEMA_VERSION", "BookRef", "Passage",
    "PdfEncrypted", "PdfUnreadable",
    "assess", "books_from_payload", "chunk_pages", "clean_page",
    "document_status", "drop_running_headers", "embed_texts", "EMBED_DIM",
    "ensure_schema", "extract_pages", "fetch_pdf", "format_passages",
    "index_pdf_bytes", "parse_doc_id", "prune_documents",
    "retrieve_for_generation", "search", "split_topics", "sync_book",
]
