"""Finding the pages of a course's books that cover a unit's topics.

Named passages rather than search because the package exports a function
called search(); a module of the same name would shadow it, and
retrieval.search would quietly mean two different things.

The syllabus line is split into its topics and each is searched for on its own,
because one embedding of five topics lands between all of them - nearest the
pages that mention several shallowly rather than the one that covers any of
them properly.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from app.config import Config

from . import embeddings, store
from .embeddings import EMBED_DIM
from .models import Passage

logger = logging.getLogger(__name__)



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
    vectors = embeddings.embed_texts(list(topics), "RETRIEVAL_QUERY")

    with store._connect() as conn:
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




def format_passages(passages: Sequence[Passage]) -> str:
    """Renders passages for the prompt, with citations the model can echo."""
    def cite(p: Passage) -> str:
        if p.page_end and p.page_end != p.page_no:
            return f"pp. {p.page_no}-{p.page_end}"
        return f"p. {p.page_no}"

    return "\n\n".join(f"[{p.book_name}, {cite(p)}]\n{p.content}" for p in passages)
