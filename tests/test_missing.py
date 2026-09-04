"""Every way a book can fail to be retrievable, against the real database.

Question generation must survive all of them by falling back to titles, which
is exactly how it behaved before RAG existed.
"""
import os
os.environ["RAG_ENABLED"] = "true"

from app import create_app
create_app()
from app import retrieval as rag
from app.retrieval import passages, store
from app.services.generation import generate_prompt

fails = []


def check(label, got, want):
    ok = got == want
    print(("  ok   " if ok else "  FAIL ") + label)
    if not ok:
        print("         got:  %r" % (got,))
        print("         want: %r" % (want,))
        fails.append(label)


def book(**kw):
    base = {"BookName": "Some Book", "BookType": "T"}
    base.update(kw)
    return base


QUERY = "Vectors, matrices, determinants"

print("books_from_payload - which entries survive")
check("FilePath missing entirely", rag.books_from_payload([book()]), [])
check("FilePath empty string", rag.books_from_payload([book(FilePath="")]), [])
check("FilePath is only whitespace", rag.books_from_payload([book(FilePath="   ")]), [])
check("FilePath is None", rag.books_from_payload([book(FilePath=None)]), [])
check("FilePath not slot_paper shaped",
      rag.books_from_payload([book(FilePath="syllabus.pdf")]), [])
check("FilePath with no extension still parses",
      [r.doc_id for r in rag.books_from_payload([book(FilePath="1_734")])], ["1_734"])
check("one usable among several unusable",
      [r.doc_id for r in rag.books_from_payload(
          [book(), book(FilePath=""), book(FilePath="2_734.pdf"), book(FilePath="x.pdf")])],
      ["2_734"])
check("empty BookDetails", rag.books_from_payload([]), [])
check("BookDetails is None", rag.books_from_payload(None), [])

print("\nretrieve_for_generation - the whole path, real database")
check("no books at all -> no passages",
      rag.retrieve_for_generation([], QUERY), [])
check("book with no FilePath -> no passages",
      rag.retrieve_for_generation([book()], QUERY), [])
check("FilePath naming a course with nothing indexed",
      rag.retrieve_for_generation([book(FilePath="1_999999.pdf")], QUERY), [])
# Indexed here rather than relying on whatever the database already holds, so
# the assertion means the same thing on any machine.
import fitz
_d = fitz.open()
_d.new_page().insert_textbox(fitz.Rect(50, 50, 550, 750),
                             "matrices determinants vectors linear algebra. " * 40,
                             fontsize=10)
rag.index_pdf_bytes(_d.tobytes(), "1_4243", 4243, 1, "Course Book", "T")

check("FilePath naming a book never indexed, on a course that HAS one",
      [p.book_name for p in rag.retrieve_for_generation(
          [book(FilePath="7_4243.pdf")], QUERY)],
      ["Course Book"])   # the search is by COURSE, so its indexed book answers

rag.prune_documents(4243, [])

print("\nthe prompt itself")
rule = {"questionType": "Short Answer", "difficultyLevel": "Easy",
        "cognitiveLevel": "Knowledge", "courseOutcome": "CO1", "mark": 5}
books = [{"BookName": "Some Book", "BookType": "T"}]

without = generate_prompt("Module 1", "Unit 1", rule, 2, books, "Vectors", "")
with_ex = generate_prompt("Module 1", "Unit 1", rule, 2, books, "Vectors", "[A Book, p.1]\ntext")

check("no passages -> no Source Extracts block at all",
      "Source Extracts" in without, False)
check("passages -> the block appears", "Source Extracts" in with_ex, True)
check("book titles are still listed either way",
      "Some Book" in without and "Some Book" in with_ex, True)
check("the two prompts differ only by that block",
      without == with_ex.replace(
          "\n    Source Extracts (verbatim from the prescribed book - "
          "base the questions on THIS text):\n[A Book, p.1]\ntext\n", ""),
      True)

no_books = generate_prompt("Module 1", "Unit 1", rule, 2, [], "Vectors", "")
check("no books -> no empty Book References heading",
      "Book References" in no_books, False)
check("no books -> the rest of the prompt is unchanged",
      "Content: \"Vectors\"" in no_books and "Module: Module 1" in no_books, True)

print("\n%d failed" % len(fails) if fails else "\nall passed")
raise SystemExit(1 if fails else 0)
