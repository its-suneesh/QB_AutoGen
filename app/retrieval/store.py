"""The database: connecting to it, shaping it, and writing a book into it.

Schema changes are numbered steps rather than one CREATE TABLE block - see
_MIGRATIONS - so a database with books already in it can move forward without
losing them.
"""

from __future__ import annotations

import logging
import re

from app.config import Config

from . import embeddings
from .embeddings import EMBED_DIM
from .pdf import assess, chunk_pages, extract_pages

logger = logging.getLogger(__name__)



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




def _same_embedding(model, dim, wanted: str | None = None) -> bool:
    """True if a stored document was embedded by the model about to be used."""
    return model == (wanted or Config.RAG_EMBED_MODEL) and dim == EMBED_DIM




def _store(doc_id, paper_id, slot_no, book_name, book_type,
           content_hash, page_count, status, quality, chunks, vectors,
           model: str | None = None) -> None:
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
             model or Config.RAG_EMBED_MODEL, EMBED_DIM),
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




def _prepare(pdf_bytes: bytes, model: str | None = None):
    """Everything derived from the PDF, before anything is written."""
    pages = extract_pages(pdf_bytes)
    status, quality = assess(pages)

    chunks, vectors = [], []
    if status == "indexed":
        chunks = chunk_pages(pages)
        if chunks:
            vectors = embeddings.embed_texts(
                [text for _, text in chunks], "RETRIEVAL_DOCUMENT", model)
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
