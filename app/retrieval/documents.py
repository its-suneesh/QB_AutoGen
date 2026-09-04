"""One book in the index: putting it there, keeping it current, taking it out.

A book is keyed by the file name the portal gives it - "1_734" - which is a ROW
POSITION, not an identity. Deleting one book renames the others, so what keeps
the index honest is the content hash and prune_documents, not the name.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Iterable

from app.config import Config

from .models import BookRef, parse_doc_id
from .pdf import PdfUnreadable
from .passages import format_passages, search, split_topics
from . import store
from .store import ensure_schema

logger = logging.getLogger(__name__)



STATUS_MESSAGES = {
    "indexed": "Indexed - this book can now be used for question generation.",
    "needs_ocr": "This PDF has no text layer (scanned). It cannot be used for "
                 "question generation until it is OCR'd.",
    "low_quality": "Text was extracted but is not readable enough to generate "
                   "reliable questions from.",
    "failed": "The PDF could not be read.",
    "stale": "Indexed by a previous embedding model - upload it again to make "
             "it searchable.",
}




def fetch_pdf(file_path: str) -> bytes:
    """
    Pulls the PDF straight from the portal's static folder. That path is served
    before the authentication middleware, so no token is needed and no backend
    change was required to make this work.
    """
    import httpx

    url = f"{Config.PORTAL_BASE_URL.rstrip('/')}/Photo/Paper/Book/{file_path}"
    response = httpx.get(url, timeout=Config.RAG_FETCH_TIMEOUT, verify=Config.RAG_VERIFY_TLS)
    response.raise_for_status()
    return response.content




def sync_book(book: BookRef) -> str:
    """
    Indexes one book if it is new or its bytes changed. Returns the status.

    The hash comparison is what makes the portal's row-based file naming safe:
    if "1_734.pdf" now holds a different book, the hash differs, the old chunks
    are dropped and the new text is indexed. Reordering self-heals on the next
    generate call.
    """
    pdf_bytes = fetch_pdf(book.file_path)
    content_hash = hashlib.sha256(pdf_bytes).hexdigest()

    with store._connect() as conn:
        row = conn.execute(
            "SELECT content_hash, book_name, status FROM book_document WHERE doc_id = %s",
            (book.doc_id,),
        ).fetchone()

    if row and row[0] == content_hash and row[1] == book.book_name:
        return row[2]                           # unchanged - nothing to do

    pages, status, quality, chunks, vectors = store._prepare(pdf_bytes)
    store._store(book.doc_id, book.paper_id, book.slot_no, book.book_name,
           book.book_type, content_hash, len(pages), status, quality,
           chunks, vectors)

    logger.info("RAG: %s (%s) -> %s", book.doc_id, book.book_name, status)
    return status




def books_from_payload(book_details: Iterable[dict]) -> list[BookRef]:
    """Turns the portal's BookDetails into BookRefs, skipping entries with no file."""
    refs: list[BookRef] = []
    for book in book_details or []:
        file_path = (book.get("FilePath") or "").strip()
        if not file_path:
            continue                      # metadata-only entry: nothing to index
        parsed = parse_doc_id(file_path)
        if not parsed:
            logger.warning("RAG: unexpected file name %r, skipping", file_path)
            continue
        doc_id, paper_id, slot_no = parsed
        refs.append(BookRef(
            doc_id=doc_id, paper_id=paper_id, slot_no=slot_no, file_path=file_path,
            book_name=book.get("BookName", ""), book_type=book.get("BookType", ""),
        ))
    return refs




def retrieve_for_generation(book_details: Iterable[dict], query: str) -> list[Passage]:
    """
    Entry point used by the generator: sync anything new, then search.

    Every failure is swallowed deliberately. If the database is unreachable or a
    PDF cannot be fetched, generation continues on titles alone exactly as it
    did before RAG existed - a retrieval problem must not stop a teacher
    producing a question paper.
    """
    if not Config.RAG_ENABLED:
        return []

    # Both checks below are free, and both are settled before anything opens a
    # connection - there is no point waiting out a database timeout to discover
    # there was nothing to look up.
    refs = books_from_payload(book_details)
    if not refs:
        return []

    topics = split_topics(query)
    if not topics:
        logger.info(
            "RAG: %r names no subject to search for, generating from titles", query
        )
        return []

    # Every book of one course carries the same paper_id, so the first is the
    # course. Say so if that ever stops being true rather than silently
    # searching only part of the shelf.
    paper_ids = {ref.paper_id for ref in refs}
    if len(paper_ids) > 1:
        logger.warning("RAG: BookDetails spans courses %s, using %s",
                       sorted(paper_ids), refs[0].paper_id)

    try:
        ensure_schema()

        # Books are normally indexed when the portal uploads them to
        # /rag/index, so there is usually nothing to do here. Pulling them
        # back from the portal stays as a fallback for books that were
        # already on disk before that endpoint existed - and only when a
        # portal address is actually configured.
        if Config.PORTAL_BASE_URL:
            for ref in refs:
                try:
                    sync_book(ref)
                except Exception:
                    logger.exception("RAG: indexing failed for %s", ref.doc_id)

        return search(refs[0].paper_id, topics)
    except Exception:
        logger.exception("RAG: retrieval unavailable, continuing without passages")
        return []




def index_pdf_bytes(
    pdf_bytes: bytes,
    doc_id: str,
    paper_id: int,
    slot_no: int,
    book_name: str,
    book_type: str = "",
    embed_model: str | None = None,
) -> dict:
    """
    Indexes a PDF supplied directly by the caller, rather than fetched from the
    portal.

    This is what the upload endpoint uses. Sending the bytes means the service
    never has to know the portal's address or reach back into it, so no
    PORTAL_BASE_URL, no network path from this service to the portal, and no
    reliance on those files staying publicly readable.

    Returns a plain dict for the HTTP response. Unlike retrieve_for_generation
    this does NOT swallow failures - the caller asked to index a specific file
    and deserves to be told what happened to it.
    """
    content_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # A caller may name the embedding model; otherwise the server's own is
    # used. Whichever it is, it is recorded on the document, because vectors
    # from different models are not comparable and a book embedded by one
    # cannot be searched with queries embedded by another.
    model = embed_model or Config.RAG_EMBED_MODEL

    ensure_schema()

    with store._connect() as conn:
        row = conn.execute(
            "SELECT content_hash, status, page_count, embed_model, embed_dim "
            "FROM book_document WHERE doc_id = %s",
            (doc_id,),
        ).fetchone()

    # Same bytes AND the same embedding model. Skipping on the bytes alone
    # would leave vectors from a previous model in place after RAG_EMBED_MODEL
    # changed - and those are not comparable to the new queries, so the book
    # would still be searched and would answer with nonsense.
    if row and row[0] == content_hash and store._same_embedding(row[3], row[4], model):
        return {
            "doc_id": doc_id, "status": row[1], "pages": row[2],
            "chunks": None, "reindexed": False,
            "message": "Already indexed - file unchanged.",
        }

    try:
        # Reading and embedding happen before anything is written, and outside
        # any connection: embedding waits out the per-minute quota, and holding
        # a connection open across that would idle out a remote Postgres.
        pages, status, quality, chunks, vectors = store._prepare(pdf_bytes, model)
    except PdfUnreadable as exc:
        # A corrupt or password-protected file is the uploader's problem to
        # fix, not a server fault - answer with a verdict, not a stack trace.
        return {
            "doc_id": doc_id, "status": "failed", "pages": 0, "chunks": 0,
            "reindexed": False,
            "message": f"{STATUS_MESSAGES['failed']} ({exc})",
        }

    chunk_count = len(chunks)
    store._store(doc_id, paper_id, slot_no, book_name, book_type,
                 content_hash, len(pages), status, quality, chunks, vectors,
                 model)

    return {
        "doc_id": doc_id,
        "status": status,
        "pages": len(pages),
        "chunks": chunk_count,
        "quality": round(quality, 3),
        "reindexed": True,
        "embed_model": model,
        "message": STATUS_MESSAGES.get(status, status),
    }




def prune_documents(paper_id: int, books) -> dict:
    """
    Makes the index match the course's current book list exactly.

    Needed because a doc_id is a ROW POSITION, not an identity. Deleting the
    second of three books renames the third from "3_734" to "2_734"; the portal
    re-sends it under the new name and it indexes correctly, but the row it used
    to occupy is left behind holding the same text. That stale row still carries
    paper_id 734, so retrieval would return the book twice - and a book removed
    from the course altogether would keep feeding questions forever.

    The portal knows the whole list whenever it saves, so it says which doc_ids
    should survive and everything else for that course goes. Chunks follow their
    document via ON DELETE CASCADE.

    Book titles are refreshed here too: renaming a book without replacing its
    PDF never reaches /rag/index, and the title is what the model cites.
    """
    keep: dict[str, tuple[str, str]] = {}
    for book in books or []:
        doc_id = (book.get("doc_id") or "").strip()
        if doc_id:
            keep[doc_id] = (book.get("BookName") or "", book.get("BookType") or "")

    # Deliberately does NOT create the schema, and does not register the vector
    # type: nothing here reads or writes an embedding, it only deletes rows and
    # refreshes titles. Requiring pgvector for that made saving a book report a
    # database-permission failure over work that never needed the extension.
    # Nothing indexed yet means nothing to prune, which is a success.
    with store._raw_connect() as conn:
        if not store._index_exists(conn):
            return {"paper_id": paper_id, "kept": 0, "removed": [], "renamed": 0}

        # The tables exist, so bring them up to date before reading them. Only
        # ensure_schema creates them, and this path deliberately does not call
        # it - but a database left on an older version would otherwise be
        # queried for columns it has not got.
        store._apply_migrations(conn, store._schema_name())

        stored = {
            row[0] for row in conn.execute(
                "SELECT doc_id FROM book_document WHERE paper_id = %s", (paper_id,)
            ).fetchall()
        }

        stale = sorted(stored - set(keep))
        if stale:
            conn.execute("DELETE FROM book_document WHERE doc_id = ANY(%s)", (stale,))
            logger.info("RAG: dropped %s from paper %s", stale, paper_id)

        renamed = 0
        for doc_id, (book_name, book_type) in keep.items():
            if doc_id not in stored:
                continue                      # not indexed yet - /rag/index will
            renamed += conn.execute(
                "UPDATE book_document SET book_name = %s, book_type = %s "
                "WHERE doc_id = %s AND (book_name IS DISTINCT FROM %s "
                "                    OR book_type IS DISTINCT FROM %s)",
                (book_name, book_type, doc_id, book_name, book_type),
            ).rowcount

    return {
        "paper_id": paper_id,
        "kept": len(stored & set(keep)),
        "removed": stale,
        "renamed": renamed,
    }




def document_status(paper_id: int) -> list[dict]:
    """Index status of every book of a course, for the portal to display."""
    # Metadata only, so no schema creation and no vector type - see the note in
    # prune_documents. A database with no index yet simply has nothing to say
    # about any book, which is an empty list rather than an error.
    with store._raw_connect() as conn:
        if not store._index_exists(conn):
            return []

        store._apply_migrations(conn, store._schema_name())

        rows = conn.execute(
            """
            SELECT d.doc_id, d.book_name, d.book_type, d.status, d.page_count,
                   d.quality, d.indexed_at, count(c.id) AS chunks,
                   d.embed_model, d.embed_dim
            FROM book_document d
            LEFT JOIN book_chunk c USING (doc_id)
            WHERE d.paper_id = %s
            GROUP BY d.doc_id
            ORDER BY d.slot_no
            """,
            (paper_id,),
        ).fetchall()

    return [
        {
            "doc_id": r[0], "book_name": r[1], "book_type": r[2],
            # A book embedded by a superseded model is reported as needing
            # re-indexing rather than as ready, because it is no longer
            # searchable even though its rows are still there.
            "status": r[3] if store._same_embedding(r[8], r[9]) else "stale",
            "pages": r[4], "quality": r[5],
            "indexed_at": r[6].isoformat() if r[6] else None,
            "chunks": r[7],
            "message": STATUS_MESSAGES.get(
                r[3] if store._same_embedding(r[8], r[9]) else "stale", r[3]),
        }
        for r in rows
    ]
