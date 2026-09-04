import os
os.environ["RAG_ENABLED"] = "true"
os.environ["RAG_DATABASE_URL"] = "postgresql://fake/fake"

from app import retrieval as rag
from app.retrieval import passages, store

fails = []


def check(label, got, want):
    ok = got == want
    print(("  ok   " if ok else "  FAIL ") + label)
    if not ok:
        print("         got:  %r" % (got,))
        print("         want: %r" % (want,))
        fails.append(label)


print("split_topics")
check("the example from the payload",
      rag.split_topics("Vectors, vector spaces, dot product, cross product, matrices."),
      ["Vectors", "vector spaces", "dot product", "cross product", "matrices"])
check("semicolons and slashes",
      rag.split_topics("Sorting; searching / hashing"),
      ["Sorting", "searching", "hashing"])
check("bullets",
      rag.split_topics("Ohm's law • Kirchhoff's laws • Thevenin"),
      ["Ohm's law", "Kirchhoff's laws", "Thevenin"])
check("dash separated",
      rag.split_topics("Limits - continuity - differentiability"),
      ["Limits", "continuity", "differentiability"])
check("one flowing sentence stays whole",
      rag.split_topics("An introduction to the theory of relativity"),
      ["An introduction to the theory of relativity"])
check("bare unit label rejected", rag.split_topics("Unit 1"), [])
check("bare module label rejected", rag.split_topics("Module 3"), [])
check("roman chapter rejected", rag.split_topics("Chapter IV"), [])
check("empty rejected", rag.split_topics(""), [])
check("None rejected", rag.split_topics(None), [])
check("label mixed with real topics is dropped",
      rag.split_topics("Unit 2, Matrices, determinants"),
      ["Matrices", "determinants"])
check("slivers dropped",
      rag.split_topics("Sets, a, relations"),
      ["Sets", "relations"])
check("long list is capped",
      len(rag.split_topics(",".join("subject%d" % i for i in range(40)))), 12)
check("\"Topic 5\" is a label, not a subject", rag.split_topics("Topic 5"), [])
check("malayalam survives",
      rag.split_topics("സംഖ്യകൾ, ഗണിതം"),
      ["സംഖ്യകൾ", "ഗണിതം"])

print("\n_merge_windows  (doc, page, distance) -> (doc, low, high, best)")
check("single hit widens by one page either side",
      passages._merge_windows([("1_734", 50, 0.20)], 1),
      [("1_734", 49, 51, 0.20)])
check("page 1 does not widen below 1",
      passages._merge_windows([("1_734", 1, 0.20)], 1),
      [("1_734", 1, 2, 0.20)])
check("overlapping hits merge into one stretch, keeping the best distance",
      passages._merge_windows([("1_734", 50, 0.30), ("1_734", 51, 0.10)], 1),
      [("1_734", 49, 52, 0.10)])
check("touching windows merge (gap of exactly one page)",
      passages._merge_windows([("1_734", 50, 0.30), ("1_734", 53, 0.40)], 1),
      [("1_734", 49, 54, 0.30)])
check("distant hits stay separate, ordered by distance",
      passages._merge_windows([("1_734", 10, 0.40), ("1_734", 90, 0.10)], 1),
      [("1_734", 89, 91, 0.10), ("1_734", 9, 11, 0.40)])
check("different books never merge",
      passages._merge_windows([("1_734", 50, 0.10), ("2_734", 50, 0.20)], 1),
      [("1_734", 49, 51, 0.10), ("2_734", 49, 51, 0.20)])
check("window 0 keeps single pages",
      passages._merge_windows([("1_734", 50, 0.10)], 0),
      [("1_734", 50, 50, 0.10)])

print("\nformat_passages")
one = rag.Passage(book_name="Linear Algebra", page_no=12, content="text", page_end=12)
many = rag.Passage(book_name="Linear Algebra", page_no=12, content="text", page_end=14)
check("single page cited as p.", rag.format_passages([one]), "[Linear Algebra, p. 12]\ntext")
check("range cited as pp.", rag.format_passages([many]), "[Linear Algebra, pp. 12-14]\ntext")
check("no passages -> empty string, so the prompt block stays out",
      rag.format_passages([]), "")

print("\n%d failed" % len(fails) if fails else "\nall passed")
raise SystemExit(1 if fails else 0)
