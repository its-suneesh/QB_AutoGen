"""End-to-end exercise of search() against a stand-in for pgvector.

The fake connection answers the two real queries by shape, so the SQL text,
the parameter order, the per-topic loop, the distance floor, the window merge
and the char budget are all exercised. Only the vector arithmetic is faked.
"""
import os
os.environ["RAG_ENABLED"] = "true"
os.environ["RAG_DATABASE_URL"] = "postgresql://fake/fake"

from app import retrieval as rag
from app.retrieval import passages, store
from app.config import Config

# A tiny "book": page -> (text, distance from each topic)
BOOK = "1_734"
PAGES = {n: "page %d text" % n for n in range(1, 101)}

# what the fake index returns per topic: (doc, page, distance)
INDEX = {
    "Vectors":       [(BOOK, 10, 0.10), (BOOK, 11, 0.15)],
    "vector spaces": [(BOOK, 11, 0.12), (BOOK, 12, 0.18)],
    "cross product": [(BOOK, 60, 0.22), (BOOK, 61, 0.30)],
    "astrophysics":  [(BOOK, 3, 0.91), (BOOK, 4, 0.95)],   # book does not cover it
}
seen_topics = []


class FakeConn:
    def execute(self, sql, params=None):
        flat = " ".join(sql.split())
        if "embedding <=>" in flat and "ORDER BY" in flat:
            topic = seen_topics.pop(0)
            # last parameter, so adding filters to the query does not shift it
            limit = params[-1]
            return _Result(INDEX[topic][:limit])
        if flat.startswith("SELECT d.book_name, c.doc_id, c.page_no, c.content"):
            # honour the OR-ed (doc_id, page BETWEEN low, high) clauses
            rows = []
            for i in range(0, len(params), 3):
                doc, low, high = params[i], params[i + 1], params[i + 2]
                for page in range(low, high + 1):
                    if page in PAGES:
                        rows.append(("Linear Algebra", doc, page, PAGES[page]))
            rows.sort(key=lambda r: (r[1], r[2]))
            return _Result(rows)
        raise AssertionError("unexpected SQL: " + flat)

    def __enter__(self): return self
    def __exit__(self, *a): return False


class _Result:
    def __init__(self, rows): self._rows = rows
    def fetchall(self): return self._rows


store._connect = lambda: FakeConn()
rag.embeddings.embed_texts = lambda texts, task, model=None: [[0.0] * 768 for _ in texts]

fails = []


def check(label, got, want):
    ok = got == want
    print(("  ok   " if ok else "  FAIL ") + label)
    if not ok:
        print("         got:  %r" % (got,))
        print("         want: %r" % (want,))
        fails.append(label)


def run(topics):
    global seen_topics
    seen_topics = list(topics)
    return rag.search(734, topics)


print("search")

# Vectors(10,11) and vector spaces(11,12) all fall inside one window -> one
# stretch; cross product(60,61) is far away -> a second.
got = run(["Vectors", "vector spaces", "cross product"])
check("adjacent topics collapse into one continuous stretch",
      [(p.page_no, p.page_end) for p in got], [(9, 13), (59, 62)])
check("closest match is offered first", got[0].page_no, 9)
check("content is the pages joined in order",
      got[0].content, "\n".join(PAGES[n] for n in range(9, 14)))
check("citation names the whole range",
      rag.format_passages(got[:1]).splitlines()[0], "[Linear Algebra, pp. 9-13]")

check("a topic the book does not cover contributes nothing",
      [(p.page_no, p.page_end) for p in run(["Vectors", "astrophysics"])],
      [(9, 12)])

check("no topic in range -> no passages at all, prompt falls back to titles",
      run(["astrophysics"]), [])

check("no topics -> no query is even attempted", rag.search(734, []), [])

Config.RAG_TOP_K = 1
check("RAG_TOP_K caps the number of stretches",
      len(run(["Vectors", "cross product"])), 1)
Config.RAG_TOP_K = 12

Config.RAG_PER_TOPIC = 1
check("RAG_PER_TOPIC limits pages taken per topic",
      [(p.page_no, p.page_end) for p in run(["Vectors", "cross product"])],
      [(9, 11), (59, 61)])
Config.RAG_PER_TOPIC = 2

Config.RAG_PAGE_WINDOW = 0
check("RAG_PAGE_WINDOW 0 sends single pages",
      [(p.page_no, p.page_end) for p in run(["Vectors"])], [(10, 11)])
Config.RAG_PAGE_WINDOW = 1

Config.RAG_MAX_CHARS = 1
got = run(["Vectors", "cross product"])
check("char budget still sends the single closest stretch", len(got), 1)
check("...and it is the closest one", got[0].page_no, 9)
Config.RAG_MAX_CHARS = 40000

Config.RAG_MAX_DISTANCE = 0.11
check("tightening the floor drops the weaker pages",
      [(p.page_no, p.page_end) for p in run(["Vectors", "vector spaces"])],
      [(9, 11)])
Config.RAG_MAX_DISTANCE = 0.55

print("\n%d failed" % len(fails) if fails else "\nall passed")
raise SystemExit(1 if fails else 0)
