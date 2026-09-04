"""Every book-management case, against a real Postgres + pgvector.

Embeddings are stubbed (free-tier quota), but the database, the schema, the
CASCADE, the prune and the hash short-circuit are all genuinely exercised.
"""
import os
os.environ["RAG_ENABLED"] = "true"
os.environ.setdefault("RAG_DATABASE_URL",
                      "postgresql://qbrag:qbrag@localhost:55432/qbrag")

import fitz
from app import retrieval as rag
from app.retrieval import passages, store

# Deterministic stand-in for the embedding API: same text -> same vector.
def fake_embed(texts, task, model=None):
    out = []
    for t in texts:
        h = abs(hash(t))
        v = [((h >> (i % 32)) & 1) * 1.0 + 0.001 * i for i in range(rag.EMBED_DIM)]
        n = sum(x * x for x in v) ** 0.5
        out.append([x / n for x in v])
    return out

rag.embeddings.embed_texts = fake_embed

PAPER = 734
fails = []


def check(label, got, want):
    ok = got == want
    print(("  ok   " if ok else "  FAIL ") + label)
    if not ok:
        print("         got:  %r" % (got,))
        print("         want: %r" % (want,))
        fails.append(label)


_PDF_CACHE = {}


def pdf(text):
    """Same text -> byte-identical PDF, so the content hash means what it says.

    fitz stamps a fresh document id on every save, so regenerating would look
    like a changed file even when the words are the same.
    """
    if text not in _PDF_CACHE:
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            # 40 repeats fits the box; more and insert_textbox silently writes
            # nothing, which the indexer would rightly call a scanned page.
            page.insert_textbox(fitz.Rect(50, 50, 550, 750),
                                ("%s page %d. " % (text, i + 1)) * 40, fontsize=10)
        _PDF_CACHE[text] = doc.tobytes()
    return _PDF_CACHE[text]


def index(slot, name, body):
    return rag.index_pdf_bytes(pdf(body), "%d_%d" % (slot, PAPER), PAPER, slot, name, "T")


def stored():
    with store._connect() as c:
        return sorted(
            (r[0], r[1]) for r in c.execute(
                "SELECT doc_id, book_name FROM book_document WHERE paper_id=%s", (PAPER,)
            ).fetchall()
        )


def chunk_docs():
    with store._connect() as c:
        return sorted(
            (r[0], r[1]) for r in c.execute(
                "SELECT doc_id, count(*) FROM book_chunk WHERE paper_id=%s GROUP BY doc_id",
                (PAPER,)
            ).fetchall()
        )


def keep(*pairs):
    return [{"doc_id": "%d_%d" % (s, PAPER), "BookName": n, "BookType": "T"} for s, n in pairs]


# clean slate
rag.ensure_schema()
with store._connect() as c:
    c.execute("DELETE FROM book_document WHERE paper_id=%s", (PAPER,))

print("three books on the shelf")
index(1, "Strang", "linear algebra vectors")
index(2, "Lay", "matrix theory")
index(3, "Axler", "abstract vector spaces")
check("all three indexed", stored(),
      [("1_734", "Strang"), ("2_734", "Lay"), ("3_734", "Axler")])
check("each has chunks", [d for d, _ in chunk_docs()], ["1_734", "2_734", "3_734"])

print("\nedit ONE book: replace 2_734's PDF, leave 1 and 3 alone")
before = dict(chunk_docs())
r = index(2, "Lay", "completely different content about determinants")
check("re-indexed", r["reindexed"], True)
check("status indexed", r["status"], "indexed")
after = dict(chunk_docs())
check("book 1 chunks untouched", after["1_734"], before["1_734"])
check("book 3 chunks untouched", after["3_734"], before["3_734"])
check("still exactly three books", len(stored()), 3)

print("\nre-send the SAME bytes for 2_734")
r = index(2, "Lay", "completely different content about determinants")
check("recognised as unchanged", r["reindexed"], False)
check("message says so", r["message"], "Already indexed - file unchanged.")

print("\nrename book 2 without touching its PDF (prune carries the new title)")
rag.prune_documents(PAPER, keep((1, "Strang"), (2, "Lay 6th edition"), (3, "Axler")))
check("title updated in the index", dict(stored())["2_734"], "Lay 6th edition")
check("nothing removed", len(stored()), 3)

print("\ndelete the middle book: 3_734 becomes 2_734")
index(2, "Axler", "abstract vector spaces")          # portal re-sends under new name
res = rag.prune_documents(PAPER, keep((1, "Strang"), (2, "Axler")))
check("stale row removed", res["removed"], ["3_734"])
check("two books left", stored(), [("1_734", "Strang"), ("2_734", "Axler")])
check("its chunks went with it (CASCADE)",
      [d for d, _ in chunk_docs()], ["1_734", "2_734"])

print("\nprune is idempotent")
res = rag.prune_documents(PAPER, keep((1, "Strang"), (2, "Axler")))
check("nothing more to remove", res["removed"], [])
check("kept count reported", res["kept"], 2)

print("\na book listed but never indexed yet")
res = rag.prune_documents(PAPER, keep((1, "Strang"), (2, "Axler"), (3, "New book")))
check("not invented as a row", len(stored()), 2)
check("and not reported removed", res["removed"], [])

print("\nremove every book from the course")
res = rag.prune_documents(PAPER, [])
check("both dropped", res["removed"], ["1_734", "2_734"])
check("index empty for this course", stored(), [])
check("no orphan chunks", chunk_docs(), [])

print("\n%d failed" % len(fails) if fails else "\nall passed")
raise SystemExit(1 if fails else 0)
