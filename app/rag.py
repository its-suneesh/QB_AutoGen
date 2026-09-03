# app/rag.py
"""
Retrieval of real textbook passages for question generation.

Why this exists
---------------
The prompt used to receive only book TITLES:

    book_references = "- Advanced Engineering Mathematics (Type: Textbook)"

so the model wrote questions from whatever it already knew about a book with
that name. The PDFs the college actually prescribes were uploaded, stored and
never read. This module opens them, indexes them once, and returns the passages
that match the unit being examined, so the generated questions come from the
prescribed text.

Design notes
------------
* Key is the file stem, e.g. "1_734"  ->  paper 734, book slot 1.
  That name is unique at any moment but NOT permanent: the portal names files
  by row position, so a delete or reorder can hand "1_734.pdf" to a different
  book. The content hash below detects exactly that and re-indexes, which is
  why nothing in the portal has to change.

* Everything fails OPEN. If Postgres is down, a PDF 404s or extraction is poor,
  the caller gets no passages and question generation proceeds exactly as it
  does today. RAG must never take the generator offline.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from app.config import Config

logger = logging.getLogger(__name__)

# --- Quality thresholds -----------------------------------------------------
# Deliberately identical to the browser-side check in the portal's textbook
# dialog, so the verdict a teacher saw at upload matches what happens here.
MIN_CHARS_PER_PAGE = 100     # below this the page carries no real text
MIN_ALPHA_RATIO = 0.55       # letters / non-space characters

CHUNK_CHARS = 3500           # ~800-900 tokens
CHUNK_OVERLAP = 500


@dataclass
class BookRef:
    doc_id: str              # "1_734"
    paper_id: int            # 734
    slot_no: int             # 1
    file_path: str           # "1_734.pdf"
    book_name: str
    book_type: str


@dataclass
class Passage:
    book_name: str
    page_no: int                  # first page of the stretch
    content: str
    page_end: int | None = None   # last page, when it spans more than one


# --------------------------------------------------------------------------
# Text extraction and cleaning
# --------------------------------------------------------------------------

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


def extract_pages(pdf_bytes: bytes) -> list[str]:
    """Per-page text, layout-aware so two-column books do not interleave."""
    import fitz  # PyMuPDF

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


def assess(pages: Sequence[str]) -> tuple[str, float]:
    """Returns (status, quality). Mirrors the portal's upload-time check."""
    if not pages:
        return "failed", 0.0

    joined = " ".join(pages)
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


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

# gemini-embedding-001 returns 3072 numbers by default. It is a Matryoshka
# model - the leading numbers carry the most meaning - so asking for 768 is a
# supported, quality-preserving truncation, not a lossy hack. 768 keeps the
# stored vectors a quarter the size and stays inside pgvector's 2000-dimension
# limit for an HNSW index, which 3072 would exceed.
_EMBED_ATTEMPTS = 4
# The quota that bites is counted per MINUTE, so the waits step up towards one.
_EMBED_BACKOFF = 20

EMBED_DIM = 768


def embed_texts(texts: Sequence[str], task: str) -> list[list[float]]:
    """
    Embeds with Gemini - the key this service already requires, so RAG adds no
    new credential. `task` is RETRIEVAL_DOCUMENT when indexing and
    RETRIEVAL_QUERY when searching; using the wrong one measurably degrades
    matching, because the two are trained as a pair.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=Config.GOOGLE_API_KEY)
    vectors: list[list[float]] = []

    config = types.EmbedContentConfig(
        task_type=task, output_dimensionality=EMBED_DIM
    )

    # Batched: one call per chunk would make indexing a textbook take minutes.
    for i in range(0, len(texts), Config.RAG_EMBED_BATCH):
        batch = list(texts[i:i + Config.RAG_EMBED_BATCH])
        result = _embed_batch(client, batch, config)
        vectors.extend([_unit(e.values) for e in result.embeddings])

    return vectors


def _embed_batch(client, batch: list[str], config):
    """
    One embedding call, retried through the per-minute quota.

    A textbook is several batches, and the embedding quota is counted per
    minute, so a long book runs into 429 partway through as a matter of course
    rather than as a fault. Without this, that would abandon the whole book and
    leave the teacher told their upload failed - so the wait is the correct
    behaviour, not a workaround.
    """
    for attempt in range(1, _EMBED_ATTEMPTS + 1):
        try:
            return client.models.embed_content(
                model=Config.RAG_EMBED_MODEL, contents=batch, config=config
            )
        except Exception as error:
            status = getattr(error, "code", None) or getattr(error, "status_code", None)
            retryable = status in (429, 500, 502, 503, 504) or "RESOURCE_EXHAUSTED" in str(error)
            if not retryable or attempt == _EMBED_ATTEMPTS:
                raise
            pause = _EMBED_BACKOFF * attempt
            logger.warning(
                "RAG: embedding quota hit, waiting %ss (attempt %s/%s)",
                pause, attempt, _EMBED_ATTEMPTS,
            )
            time.sleep(pause)


def _unit(values: Sequence[float]) -> list[float]:
    """
    Scales a vector to length 1.

    Only the full-width output of gemini-embedding-001 arrives normalised; a
    truncated one does not. Cosine distance would cope either way, but storing
    unit vectors keeps the distances comparable between batches and leaves the
    door open to an inner-product index, which does NOT cope.
    """
    length = math.sqrt(sum(v * v for v in values))
    return [v / length for v in values] if length else list(values)


# --------------------------------------------------------------------------
# pgvector store
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Schema, and how it changes
# --------------------------------------------------------------------------
#
# CREATE TABLE IF NOT EXISTS is not enough on its own: it does nothing to a
# table that already exists, so adding a column later would silently be skipped
# and the code would then query a column that is not there. The alternative -
# dropping and rebuilding - throws away every embedding, and re-embedding a
# shelf of textbooks costs real API quota and an hour of waiting.
#
# So each change is a numbered step, applied once, in order, and recorded. A
# database at any version reaches the current one without losing a row.
#
# To change the schema: append a step with the next number. Never edit a step
# that has shipped - a database that already ran it will not run it again.

SCHEMA_VERSION = 2

_MIGRATIONS: list[tuple[int, tuple[str, ...]]] = [
    (1, (
        """
        CREATE TABLE IF NOT EXISTS book_document (
            doc_id       text PRIMARY KEY,
            paper_id     integer NOT NULL,
            slot_no      integer NOT NULL,
            book_name    text,
            book_type    text,
            content_hash text NOT NULL,
            page_count   integer,
            status       text NOT NULL,
            quality      real,
            indexed_at   timestamptz NOT NULL DEFAULT now()
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS book_chunk (
            id        bigserial PRIMARY KEY,
            doc_id    text NOT NULL REFERENCES book_document(doc_id) ON DELETE CASCADE,
            paper_id  integer NOT NULL,
            page_no   integer,
            content   text NOT NULL,
            embedding vector({EMBED_DIM})
        )
        """,
        "CREATE INDEX IF NOT EXISTS book_chunk_paper_idx ON book_chunk (paper_id)",
        "CREATE INDEX IF NOT EXISTS book_chunk_vec_idx "
        "ON book_chunk USING hnsw (embedding vector_cosine_ops)",
    )),

    # Which model produced each document's vectors.
    #
    # Vectors from different models are not comparable - distances between them
    # are meaningless rather than merely worse - so a book embedded by an older
    # model has to be re-indexed, not searched alongside the new ones. Without
    # this recorded, switching RAG_EMBED_MODEL would leave the old vectors in
    # place and quietly return nonsense, because the content hash still matches
    # and indexing would skip the book entirely.
    (2, (
        "ALTER TABLE book_document ADD COLUMN IF NOT EXISTS embed_model text",
        "ALTER TABLE book_document ADD COLUMN IF NOT EXISTS embed_dim integer",
        # Rows that predate this column were embedded by whatever was
        # configured then; naming it here would be a guess, so they are left
        # NULL and treated as needing a re-index.
    )),
]


def _apply_migrations(conn, schema: str) -> None:
    """Brings the database up to SCHEMA_VERSION, one recorded step at a time."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version ("
        "  version    integer NOT NULL,"
        "  applied_at timestamptz NOT NULL DEFAULT now()"
        ")"
    )

    row = conn.execute("SELECT max(version) FROM schema_version").fetchone()
    current = row[0] if row and row[0] is not None else 0

    # A database built before this table existed already carries step 1's
    # tables. Recording that rather than re-running it keeps the step's own
    # IF NOT EXISTS from being the thing that saves us.
    if current == 0 and _index_exists(conn):
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")
        current = 1
        logger.info("RAG: adopted existing tables in %s as schema version 1", schema)

    for version, statements in _MIGRATIONS:
        if version <= current:
            continue
        # One transaction per step: a step that fails halfway leaves the
        # database on the version it was already working at, not between two.
        with conn.transaction():
            for statement in statements:
                conn.execute(statement)
            conn.execute("INSERT INTO schema_version (version) VALUES (%s)", (version,))
        logger.info("RAG: applied schema version %s in %s", version, schema)


def _connect():
    """
    Opens a connection that is guaranteed to give up quickly.

    Both timeouts matter more here than they look. retrieve_for_generation runs
    inside the 180s ceiling that covers the whole generate request, so a
    database that is merely UNREACHABLE - a stopped container, a hostname that
    no longer resolves - would otherwise block until libpq's own default gave
    up, and turn "generation continues without passages" into a 504 for the
    teacher. Failing open has to mean failing FAST.
    """
    from pgvector.psycopg import register_vector

    conn = _raw_connect()
    register_vector(conn)
    return conn


def _raw_connect():
    """
    A connection that has NOT registered the vector type.

    ensure_schema needs this: registering the type looks it up in the database,
    which fails on a database where CREATE EXTENSION has not run yet - and that
    statement lives in the very schema this is opening the connection to create.
    Everything after the schema exists uses _connect instead.
    """
    import psycopg

    # search_path puts the service's own schema first and keeps public behind
    # it, because the vector TYPE belongs to the extension and lives there.
    # Every table name in this module is unqualified and resolves through it.
    return psycopg.connect(
        Config.RAG_DATABASE_URL,
        autocommit=True,
        connect_timeout=Config.RAG_DB_CONNECT_TIMEOUT,
        options=(
            f"-c statement_timeout={Config.RAG_DB_STATEMENT_TIMEOUT * 1000} "
            f"-c search_path={_schema_name()},public"
        ),
    )


def _schema_name() -> str:
    """
    The configured schema, refused unless it is a plain identifier.

    It is interpolated into SQL that cannot take a parameter - search_path and
    CREATE SCHEMA both name it - so it is checked here rather than trusted.
    """
    name = (Config.RAG_DB_SCHEMA or "qbrag").strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"RAG_DB_SCHEMA is not a valid schema name: {name!r}")
    return name


def ensure_schema() -> None:
    schema = _schema_name()
    with _raw_connect() as conn:
        _ensure_extension(conn)
        try:
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            _apply_migrations(conn, schema)
        except Exception as error:
            if "permission denied" not in str(error):
                raise
            # Creating a schema needs CREATE on the DATABASE, which an ordinary
            # login usually has and which is a far smaller ask than rights on
            # public. If even that is refused there is nothing left to try.
            raise RuntimeError(
                f"This login may not create the {schema!r} schema it needs. A "
                "superuser needs to run, once:  GRANT CREATE ON DATABASE "
                f"<database> TO {_current_user(conn)};  (original error: {error})"
            ) from error


def _current_user(conn) -> str:
    try:
        return conn.execute("SELECT current_user").fetchone()[0]
    except Exception:
        return "the application user"


def _ensure_extension(conn) -> None:
    """
    Makes sure the vector type exists, and says something useful if it cannot.

    CREATE EXTENSION needs rights an ordinary application login usually does
    not have on a shared or managed Postgres. It is checked first so that the
    normal case - already enabled - never needs those rights at all, and so a
    refusal reports the one command an administrator has to run instead of a
    bare "permission denied" from three frames deep in an upload.
    """
    if conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone():
        return

    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception as error:
        raise RuntimeError(
            "The pgvector extension is not enabled on this database and this "
            "login may not enable it. A superuser needs to run, once:  "
            "CREATE EXTENSION vector;  "
            f"(original error: {error})"
        ) from error


UPSERT_DOCUMENT_SQL = """
INSERT INTO book_document
    (doc_id, paper_id, slot_no, book_name, book_type,
     content_hash, page_count, status, quality, indexed_at,
     embed_model, embed_dim)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, now(), %s, %s)
ON CONFLICT (doc_id) DO UPDATE SET
    paper_id = EXCLUDED.paper_id, slot_no = EXCLUDED.slot_no,
    book_name = EXCLUDED.book_name, book_type = EXCLUDED.book_type,
    content_hash = EXCLUDED.content_hash, page_count = EXCLUDED.page_count,
    status = EXCLUDED.status, quality = EXCLUDED.quality, indexed_at = now(),
    embed_model = EXCLUDED.embed_model, embed_dim = EXCLUDED.embed_dim
"""


def _same_embedding(model, dim) -> bool:
    """True if a stored document was embedded by the model in use now."""
    return model == Config.RAG_EMBED_MODEL and dim == EMBED_DIM


def _store(doc_id, paper_id, slot_no, book_name, book_type,
           content_hash, page_count, status, quality, chunks, vectors) -> None:
    """
    Writes one book's document row and its chunks.

    The order is not cosmetic: book_chunk.doc_id is a foreign key onto
    book_document, so the document row has to exist before any chunk referring
    to it. Getting this backwards fails only on a real database and only for a
    book being indexed for the FIRST time, which is the easiest case to believe
    already works.

    Everything that can fail - reading the PDF, embedding it - is done by the
    caller before this is reached, so a book is never left recorded as indexed
    with no text behind it.
    """
    with _connect() as conn:
        conn.execute(
            UPSERT_DOCUMENT_SQL,
            (doc_id, paper_id, slot_no, book_name, book_type,
             content_hash, page_count, status, quality,
             Config.RAG_EMBED_MODEL, EMBED_DIM),
        )
        conn.execute("DELETE FROM book_chunk WHERE doc_id = %s", (doc_id,))

        if chunks:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO book_chunk (doc_id, paper_id, page_no, content, embedding)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    [
                        (doc_id, paper_id, page_no, text, vector)
                        for (page_no, text), vector in zip(chunks, vectors)
                    ],
                )


def _prepare(pdf_bytes: bytes):
    """Everything derived from the PDF, before anything is written."""
    pages = extract_pages(pdf_bytes)
    status, quality = assess(pages)

    chunks, vectors = [], []
    if status == "indexed":
        chunks = chunk_pages(pages)
        if chunks:
            vectors = embed_texts([text for _, text in chunks], "RETRIEVAL_DOCUMENT")
        else:
            status = "low_quality"

    return pages, status, quality, chunks, vectors


def _index_exists(conn) -> bool:
    """True if the index tables have been created on this database."""
    # Unqualified, so it resolves through search_path to whichever schema the
    # service is configured to use.
    return conn.execute(
        "SELECT to_regclass('book_document') IS NOT NULL"
    ).fetchone()[0]


def parse_doc_id(file_path: str) -> tuple[str, int, int] | None:
    """'1_734.pdf' -> ('1_734', 734, 1). Returns None if it isn't that shape."""
    stem = Path(file_path).stem
    match = re.fullmatch(r"(\d+)_(\d+)", stem)
    if not match:
        return None
    slot_no, paper_id = int(match.group(1)), int(match.group(2))
    return stem, paper_id, slot_no


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

    with _connect() as conn:
        row = conn.execute(
            "SELECT content_hash, book_name, status FROM book_document WHERE doc_id = %s",
            (book.doc_id,),
        ).fetchone()

    if row and row[0] == content_hash and row[1] == book.book_name:
        return row[2]                           # unchanged - nothing to do

    pages, status, quality, chunks, vectors = _prepare(pdf_bytes)
    _store(book.doc_id, book.paper_id, book.slot_no, book.book_name,
           book.book_type, content_hash, len(pages), status, quality,
           chunks, vectors)

    logger.info("RAG: %s (%s) -> %s", book.doc_id, book.book_name, status)
    return status


# A syllabus line is a LIST of topics, not one subject:
#
#     "Vectors, vector spaces, dot product, cross product, matrices."
#
# Embedded whole, that produces the centroid of five topics - a point nearest
# the pages that mention several of them shallowly (chapter openers, summaries,
# exercise lists) and far from the page that covers any one of them properly.
# Question generation wants the opposite, so each topic is searched separately.
_TOPIC_SPLIT = re.compile(r"[,;/\n•]+|\s+[-–]\s+")

# "Unit 1", "Module 3", "Chapter IV" - a position in the syllabus, not a
# subject. No page of a textbook is meaningfully near it.
_LABEL_ONLY = re.compile(
    r"^\s*(unit|module|chapter|part|section|topic)\s*[-:.)]?\s*[0-9ivxlIVXL]*\s*$",
    re.IGNORECASE,
)

# Bounds the embedding call for a pathologically long syllabus line.
_MAX_TOPICS = 12


def split_topics(content: str) -> list[str]:
    """
    Splits a syllabus line into the topics it lists.

    Returns [] when the text names no subject at all - a bare "Unit 1", or an
    empty description. Searching on that would return arbitrary pages, and the
    prompt instructs the model to base its questions on whatever it is handed,
    so arbitrary pages are actively worse than none.
    """
    text = (content or "").strip()
    if not text or _LABEL_ONLY.match(text):
        return []

    topics = [
        part for part in (p.strip(" .\t") for p in _TOPIC_SPLIT.split(text))
        if len(part) >= 3 and not _LABEL_ONLY.match(part) and re.search(r"\w", part)
    ]

    # One flowing sentence rather than a list: search it whole.
    if len(topics) < 2:
        return [text]
    return topics[:_MAX_TOPICS]


def _merge_windows(
    hits: Sequence[tuple[str, int, float]], window: int
) -> list[tuple[str, int, int, float]]:
    """
    Widens each hit to the pages around it and merges the ones that touch.

    Two topics of the same unit usually land on neighbouring pages; without the
    merge they would be sent as two overlapping passages, paying twice for the
    same text and repeating it to the model.
    """
    by_doc: dict[str, list[tuple[int, int, float]]] = {}
    for doc_id, page_no, distance in hits:
        by_doc.setdefault(doc_id, []).append(
            (max(1, page_no - window), page_no + window, distance)
        )

    ranges: list[tuple[str, int, int, float]] = []
    for doc_id, spans in by_doc.items():
        spans.sort()
        low, high, best = spans[0]
        for next_low, next_high, distance in spans[1:]:
            if next_low <= high + 1:                  # touching: one stretch
                high = max(high, next_high)
                best = min(best, distance)
            else:
                ranges.append((doc_id, low, high, best))
                low, high, best = next_low, next_high, distance
        ranges.append((doc_id, low, high, best))

    ranges.sort(key=lambda r: r[3])                   # closest match first
    return ranges


def _passages_for(conn, ranges: Sequence[tuple[str, int, int, float]]) -> list[Passage]:
    """Reads each page range back as one continuous passage."""
    clauses, params = [], []
    for doc_id, low, high, _ in ranges:
        clauses.append("(c.doc_id = %s AND c.page_no BETWEEN %s AND %s)")
        params += [doc_id, low, high]

    rows = conn.execute(
        "SELECT d.book_name, c.doc_id, c.page_no, c.content "
        "FROM book_chunk c JOIN book_document d USING (doc_id) "
        "WHERE " + " OR ".join(clauses) + " "
        "ORDER BY c.doc_id, c.page_no, c.id",
        params,
    ).fetchall()

    pages_of: dict[str, list[tuple[int, str, str]]] = {}
    for book_name, doc_id, page_no, content in rows:
        pages_of.setdefault(doc_id, []).append((page_no, content, book_name))

    passages: list[Passage] = []
    budget = Config.RAG_MAX_CHARS
    for doc_id, low, high, _ in ranges:
        pieces = [p for p in pages_of.get(doc_id, []) if low <= p[0] <= high]
        if not pieces:
            continue

        body = "\n".join(p[1] for p in pieces)
        # The closest match is always sent, however long, because dropping it
        # would leave the model with only the weaker passages.
        if passages and len(body) > budget:
            break
        budget -= len(body)

        passages.append(Passage(
            book_name=pieces[0][2],
            page_no=pieces[0][0],
            page_end=pieces[-1][0],
            content=body,
        ))

    return passages


def search(paper_id: int, topics: Sequence[str], limit: int | None = None) -> list[Passage]:
    """
    The pages of this course's books covering the given topics.

    Each topic is searched for on its own and every hit widened to the pages
    around it, so what reaches the model is a handful of readable stretches of
    the book spread across the whole unit - rather than scattered paragraphs
    clustered around one point of it.
    """
    if not topics:
        return []

    limit = limit or Config.RAG_TOP_K
    vectors = embed_texts(list(topics), "RETRIEVAL_QUERY")

    with _connect() as conn:
        hits: list[tuple[str, int, float]] = []
        for topic, vector in zip(topics, vectors):
            rows = conn.execute(
                """
                SELECT c.doc_id, c.page_no, c.embedding <=> %s::vector AS distance
                FROM book_chunk c
                JOIN book_document d USING (doc_id)
                WHERE c.paper_id = %s AND d.status = 'indexed'
                  AND d.embed_model = %s AND d.embed_dim = %s
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (vector, paper_id, Config.RAG_EMBED_MODEL, EMBED_DIM,
                 vector, Config.RAG_PER_TOPIC),
            ).fetchall()

            near = [
                (r[0], r[1], r[2]) for r in rows
                if r[1] is not None and r[2] <= Config.RAG_MAX_DISTANCE
            ]
            if not near:
                # The prescribed book does not cover this topic. Sending its
                # nearest pages anyway would pull the questions off-syllabus.
                #
                # The distance that just missed is logged on purpose: it is the
                # only way to tell a floor set too tight (real pages being
                # thrown away, and the fallback to titles is silent) from a book
                # that genuinely does not cover the unit. Tune RAG_MAX_DISTANCE
                # from these numbers rather than by guessing.
                nearest = f"{rows[0][2]:.3f}" if rows else "no pages indexed"
                logger.info(
                    "RAG: no page covers %r - nearest was %s, floor is %.2f",
                    topic, nearest, Config.RAG_MAX_DISTANCE,
                )
            hits += near

        if not hits:
            return []

        return _passages_for(conn, _merge_windows(hits, Config.RAG_PAGE_WINDOW)[:limit])


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


def format_passages(passages: Sequence[Passage]) -> str:
    """Renders passages for the prompt, with citations the model can echo."""
    def cite(p: Passage) -> str:
        if p.page_end and p.page_end != p.page_no:
            return f"pp. {p.page_no}-{p.page_end}"
        return f"p. {p.page_no}"

    return "\n\n".join(f"[{p.book_name}, {cite(p)}]\n{p.content}" for p in passages)


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


def index_pdf_bytes(
    pdf_bytes: bytes,
    doc_id: str,
    paper_id: int,
    slot_no: int,
    book_name: str,
    book_type: str = "",
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
    ensure_schema()

    with _connect() as conn:
        row = conn.execute(
            "SELECT content_hash, status, page_count, embed_model, embed_dim "
            "FROM book_document WHERE doc_id = %s",
            (doc_id,),
        ).fetchone()

    # Same bytes AND the same embedding model. Skipping on the bytes alone
    # would leave vectors from a previous model in place after RAG_EMBED_MODEL
    # changed - and those are not comparable to the new queries, so the book
    # would still be searched and would answer with nonsense.
    if row and row[0] == content_hash and _same_embedding(row[3], row[4]):
        return {
            "doc_id": doc_id, "status": row[1], "pages": row[2],
            "chunks": None, "reindexed": False,
            "message": "Already indexed - file unchanged.",
        }

    try:
        # Reading and embedding happen before anything is written, and outside
        # any connection: embedding waits out the per-minute quota, and holding
        # a connection open across that would idle out a remote Postgres.
        pages, status, quality, chunks, vectors = _prepare(pdf_bytes)
    except PdfUnreadable as exc:
        # A corrupt or password-protected file is the uploader's problem to
        # fix, not a server fault - answer with a verdict, not a stack trace.
        return {
            "doc_id": doc_id, "status": "failed", "pages": 0, "chunks": 0,
            "reindexed": False,
            "message": f"{STATUS_MESSAGES['failed']} ({exc})",
        }

    chunk_count = len(chunks)
    _store(doc_id, paper_id, slot_no, book_name, book_type,
           content_hash, len(pages), status, quality, chunks, vectors)

    return {
        "doc_id": doc_id,
        "status": status,
        "pages": len(pages),
        "chunks": chunk_count,
        "quality": round(quality, 3),
        "reindexed": True,
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
    with _raw_connect() as conn:
        if not _index_exists(conn):
            return {"paper_id": paper_id, "kept": 0, "removed": [], "renamed": 0}

        # The tables exist, so bring them up to date before reading them. Only
        # ensure_schema creates them, and this path deliberately does not call
        # it - but a database left on an older version would otherwise be
        # queried for columns it has not got.
        _apply_migrations(conn, _schema_name())

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
    with _raw_connect() as conn:
        if not _index_exists(conn):
            return []

        _apply_migrations(conn, _schema_name())

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
            "status": r[3] if _same_embedding(r[8], r[9]) else "stale",
            "pages": r[4], "quality": r[5],
            "indexed_at": r[6].isoformat() if r[6] else None,
            "chunks": r[7],
            "message": STATUS_MESSAGES.get(
                r[3] if _same_embedding(r[8], r[9]) else "stale", r[3]),
        }
        for r in rows
    ]
