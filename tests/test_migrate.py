"""Schema changes must not cost data, and a model change must not go unnoticed.

Runs against a real Postgres. Builds a database at the OLD shape with rows in
it, then upgrades and checks the rows are still there.
"""
import os
os.environ["RAG_ENABLED"] = "true"
os.environ.setdefault("RAG_DATABASE_URL",
                      "postgresql://qbrag:qbrag@localhost:55432/qbrag")

import fitz
from app import create_app
create_app()
from app import retrieval as rag
from app.retrieval import passages, store
from app.config import Config

Config.RAG_DATABASE_URL = os.environ["RAG_DATABASE_URL"]
SCHEMA = store._schema_name()
PAPER = 4242
fails = []


def check(label, got, want):
    ok = got == want
    print(("  ok   " if ok else "  FAIL ") + label)
    if not ok:
        print("         got:  %r" % (got,))
        print("         want: %r" % (want,))
        fails.append(label)


def fake_embed(texts, task, model=None):
    out = []
    for t in texts:
        h = abs(hash(t))
        v = [((h >> (i % 32)) & 1) * 1.0 + 0.001 * i for i in range(rag.EMBED_DIM)]
        n = sum(x * x for x in v) ** 0.5
        out.append([x / n for x in v])
    return out


rag.embeddings.embed_texts = fake_embed

_PDF = {}


def pdf(text):
    if text not in _PDF:
        d = fitz.open()
        d.new_page().insert_textbox(fitz.Rect(50, 50, 550, 750),
                                    ("%s. " % text) * 40, fontsize=10)
        _PDF[text] = d.tobytes()
    return _PDF[text]


def sql(*statements):
    with store._raw_connect() as c:
        for st in statements:
            c.execute(st)


def one(query, params=None):
    with store._raw_connect() as c:
        r = c.execute(query, params).fetchone()
        return r[0] if r else None


print("start from a database at the OLD shape, with data in it")
sql(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE", f"CREATE SCHEMA {SCHEMA}")
# exactly what version 1 created - no embed_model column, no schema_version
sql(
    """CREATE TABLE book_document (
         doc_id text PRIMARY KEY, paper_id integer NOT NULL, slot_no integer NOT NULL,
         book_name text, book_type text, content_hash text NOT NULL, page_count integer,
         status text NOT NULL, quality real,
         indexed_at timestamptz NOT NULL DEFAULT now())""",
    f"""CREATE TABLE book_chunk (
         id bigserial PRIMARY KEY,
         doc_id text NOT NULL REFERENCES book_document(doc_id) ON DELETE CASCADE,
         paper_id integer NOT NULL, page_no integer, content text NOT NULL,
         embedding vector({rag.EMBED_DIM}))""",
    "INSERT INTO book_document (doc_id, paper_id, slot_no, book_name, book_type,"
    " content_hash, page_count, status, quality)"
    f" VALUES ('1_{PAPER}', {PAPER}, 1, 'Precious Book', 'T', 'abc123', 300, 'indexed', 0.95)",
    "INSERT INTO book_chunk (doc_id, paper_id, page_no, content, embedding)"
    f" VALUES ('1_{PAPER}', {PAPER}, 7, 'irreplaceable text', "
    f"'{[0.1] * rag.EMBED_DIM}')",
)
check("old-shape row present", one("SELECT book_name FROM book_document"), "Precious Book")
check("no version table yet",
      one("SELECT to_regclass('schema_version') IS NULL"), True)
check("no embed_model column yet", one(
    "SELECT count(*)=0 FROM information_schema.columns "
    f"WHERE table_schema='{SCHEMA}' AND table_name='book_document' "
    "AND column_name='embed_model'"), True)

print("\nupgrade")
rag.ensure_schema()
check("reached the current version",
      one("SELECT max(version) FROM schema_version"), rag.SCHEMA_VERSION)
check("adopted the existing tables as version 1, did not rebuild them",
      one("SELECT count(*) FROM schema_version"), 2)
check("THE DATA IS STILL THERE",
      one("SELECT book_name FROM book_document"), "Precious Book")
check("its chunk is still there",
      one("SELECT content FROM book_chunk"), "irreplaceable text")
check("page number preserved", one("SELECT page_no FROM book_chunk"), 7)
check("new column exists", one(
    "SELECT count(*)=1 FROM information_schema.columns "
    f"WHERE table_schema='{SCHEMA}' AND table_name='book_document' "
    "AND column_name='embed_model'"), True)
check("pre-existing row has no model recorded, as it cannot be known",
      one("SELECT embed_model FROM book_document"), None)

print("\nrunning it again changes nothing")
rag.ensure_schema()
check("still at the same version",
      one("SELECT max(version) FROM schema_version"), rag.SCHEMA_VERSION)
check("no duplicate version rows",
      one("SELECT count(*) FROM schema_version"), 2)
check("data untouched", one("SELECT book_name FROM book_document"), "Precious Book")

print("\na book indexed by an older model is not searched, and says so")
status = rag.document_status(PAPER)
check("reported as stale", status[0]["status"], "stale")
check("message tells the user what to do",
      "upload it again" in status[0]["message"], True)
check("excluded from search",
      rag.search(PAPER, ["irreplaceable text"]), [])

print("\nre-indexing it repairs the row")
r = rag.index_pdf_bytes(pdf("irreplaceable text about vectors"),
                        f"1_{PAPER}", PAPER, 1, "Precious Book", "T")
check("re-indexed", r["status"], "indexed")
check("model now recorded",
      one("SELECT embed_model FROM book_document"), Config.RAG_EMBED_MODEL)
check("dimension recorded", one("SELECT embed_dim FROM book_document"), rag.EMBED_DIM)
check("searchable again", len(rag.search(PAPER, ["vectors"])) > 0, True)

print("\nsame bytes, different model -> re-indexed rather than skipped")
before = Config.RAG_EMBED_MODEL
r = rag.index_pdf_bytes(pdf("irreplaceable text about vectors"),
                        f"1_{PAPER}", PAPER, 1, "Precious Book", "T")
check("unchanged file is skipped normally", r["reindexed"], False)
Config.RAG_EMBED_MODEL = "models/some-newer-model"
r = rag.index_pdf_bytes(pdf("irreplaceable text about vectors"),
                        f"1_{PAPER}", PAPER, 1, "Precious Book", "T")
check("model change forces a re-index despite identical bytes", r["reindexed"], True)
check("new model recorded",
      one("SELECT embed_model FROM book_document"), "models/some-newer-model")
Config.RAG_EMBED_MODEL = before

sql(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
print("\n%d failed" % len(fails) if fails else "\nall passed")
raise SystemExit(1 if fails else 0)
