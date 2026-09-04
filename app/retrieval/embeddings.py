"""Turning text into vectors, and surviving the quota while doing it."""

from __future__ import annotations

import logging
import math
import time
from typing import Sequence

from app import usage
from app.config import Config

logger = logging.getLogger(__name__)



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




def embed_texts(texts: Sequence[str], task: str,
                model: str | None = None) -> list[list[float]]:
    """
    Embeds with Gemini - the key this service already requires, so RAG adds no
    new credential. `task` is RETRIEVAL_DOCUMENT when indexing and
    RETRIEVAL_QUERY when searching; using the wrong one measurably degrades
    matching, because the two are trained as a pair.
    """
    from google import genai
    from google.genai import types

    # The caller's model when it named one, the server's otherwise. Both the
    # documents and the queries must use the SAME one - vectors from different
    # models are not comparable - which is why it is recorded per document.
    model = model or Config.RAG_EMBED_MODEL

    client = genai.Client(api_key=Config.GOOGLE_API_KEY)
    vectors: list[list[float]] = []

    config = types.EmbedContentConfig(
        task_type=task, output_dimensionality=EMBED_DIM
    )

    # Batched: one call per chunk would make indexing a textbook take minutes.
    for i in range(0, len(texts), Config.RAG_EMBED_BATCH):
        batch = list(texts[i:i + Config.RAG_EMBED_BATCH])
        result = _embed_batch(client, batch, config, model)
        usage.record_embedding(model, batch, tokens=_count_tokens(client, batch, model))
        vectors.extend([_unit(e.values) for e in result.embeddings])

    return vectors




def _count_tokens(client, batch: list[str], model: str) -> int | None:
    """
    How many tokens a batch is, asked of the model that is about to embed it.

    The embedding endpoint reports nothing to count by: on the Gemini Developer
    API both `metadata` and each embedding's `statistics` come back None, so the
    only figures available from the call itself are the ones we already had -
    how many texts were sent and how long they were.

    count_tokens answers with the model's own tokenizer, is not billed, and at
    RAG_EMBED_BATCH=100 costs one extra round trip per hundred chunks - a whole
    textbook is typically one. That is worth it for a real number in place of a
    null in a field labelled "tokens".

    Never allowed to fail the indexing. A book that is embedded but unmetered is
    a missing figure; a book that is not embedded because the meter broke is a
    missing book.
    """
    try:
        return client.models.count_tokens(model=model, contents=batch).total_tokens
    except Exception:
        logger.warning("RAG: token count unavailable for this batch", exc_info=True)
        return None




def _embed_batch(client, batch: list[str], config, model: str):
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
                model=model, contents=batch, config=config
            )
        except Exception as error:
            status = getattr(error, "code", None) or getattr(error, "status_code", None)
            retryable = status in (429, 500, 502, 503, 504) or "RESOURCE_EXHAUSTED" in str(error)
            if not retryable or attempt == _EMBED_ATTEMPTS:
                raise
            usage.record_rate_limit()
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
