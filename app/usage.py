"""What one request cost: which model answered, and how much of it was spent.

Every AI request here fans out - generation makes one call per rule, extraction
one per batch of questions, indexing one per batch of chunks - so the number
that matters is the TOTAL for the request, not for any single call. The counts
are gathered in a context variable rather than returned up the stack, because
the calls run concurrently under asyncio.gather and threading the totals back
through every layer would touch code that has nothing else to do with billing.

Reported back to the portal, not only logged. A number in a log file answers
"why was the bill high last month"; a number in the response answers "what did
THIS cost", which is the question someone actually asks while using the thing.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_current: contextvars.ContextVar["Report | None"] = contextvars.ContextVar(
    "usage_report", default=None
)


@dataclass
class Report:
    """Everything one request spent, across every call it made."""

    provider: str = ""          # claude | gemini
    model: str = ""             # claude-sonnet-5, models/gemini-embedding-001
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_written: int = 0      # billed at ~1.25x
    cache_read: int = 0         # billed at ~0.1x
    rate_limited: int = 0       # 429s waited out and retried

    # The embedding endpoint reports nothing to count by, so the tokens are
    # asked for separately with count_tokens - the model's own tokenizer, not a
    # guess. None when that ask failed, which must not be read as zero.
    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_calls: int = 0
    embedded_texts: int = 0
    embedded_characters: int = 0
    embedded_tokens: int | None = None
    tokens_uncounted: bool = False   # some batch could not be counted

    def as_dict(self) -> dict:
        """
        What the response carries: who answered, and how much it cost.

        Four keys, the same four everywhere, so a caller never has to work out
        which endpoint a usage block came from. Indexing a book only embeds, so
        there the model IS the embedding model.

        The rest of what is collected - cache hits, 429s, call counts - stays in
        the log. It answers "why was last month expensive", which is a question
        asked of a month, not of one response.

        Indexing spends input tokens only - an embedding call answers with a
        vector, so its output is 0 as a fact rather than as a missing figure.
        input_tokens stays null only when count_tokens could not be reached, so
        a null here means unmeasured and never means free.
        """
        if self.embedded_texts and not self.calls:
            return {
                "provider": self.embedding_provider,
                "model": self.embedding_model,
                "input_tokens": self.embedded_tokens,
                "output_tokens": 0,
            }

        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@contextlib.contextmanager
def collect():
    """Gathers everything spent inside the block into one report."""
    report = Report()
    token = _current.set(report)
    try:
        yield report
    finally:
        _current.reset(token)

        # Reported differently because they ARE different: an embedding call
        # returns no token counts, so printing token fields for one would show
        # a row of zeros that reads like a failure.
        limited = f" | {report.rate_limited} rate-limited" if report.rate_limited else ""

        if report.calls:
            logger.info(
                "usage: %s/%s | %d call(s) | in %d out %d | cache %d written %d read%s",
                report.provider, report.model, report.calls,
                report.input_tokens, report.output_tokens,
                report.cache_written, report.cache_read, limited,
            )
        if report.embedded_texts:
            logger.info(
                "usage: %s | %d text(s), %d chars, %s tokens embedded%s",
                report.embedding_model, report.embedded_texts,
                report.embedded_characters,
                report.embedded_tokens if report.embedded_tokens is not None else "?",
                limited,
            )


def record_llm(provider: str, model: str, response) -> None:
    """
    Adds one model call to the current report.

    The two SDKs name the same numbers differently, which is the only reason
    this is not two lines at the call site.
    """
    report = _current.get()
    if report is None:
        return

    report.provider = provider
    report.model = model
    report.calls += 1

    if provider == "claude":
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        report.input_tokens += getattr(usage, "input_tokens", 0) or 0
        report.output_tokens += getattr(usage, "output_tokens", 0) or 0
        report.cache_written += getattr(usage, "cache_creation_input_tokens", 0) or 0
        report.cache_read += getattr(usage, "cache_read_input_tokens", 0) or 0
    else:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        report.input_tokens += getattr(usage, "prompt_token_count", 0) or 0
        report.output_tokens += getattr(usage, "candidates_token_count", 0) or 0
        report.cache_read += getattr(usage, "cached_content_token_count", 0) or 0


def record_embedding(model: str, texts: list, provider: str = "gemini",
                     tokens: int | None = None) -> None:
    """
    Adds one embedding call.

    `tokens` comes from count_tokens at the call site, because the embedding
    response itself carries no usage of any kind. It is None when that ask
    failed, and a batch that could not be counted must leave the request's total
    None rather than quietly under-report it - a partial sum in a field labelled
    "tokens" is worse than an admitted gap.
    """
    report = _current.get()
    if report is None:
        return

    report.embedding_provider = provider
    report.embedding_model = model
    report.embedding_calls += 1
    report.embedded_texts += len(texts)
    report.embedded_characters += sum(len(t) for t in texts)

    if tokens is None:
        report.tokens_uncounted = True

    if report.tokens_uncounted:
        report.embedded_tokens = None
    else:
        report.embedded_tokens = (report.embedded_tokens or 0) + tokens


def record_rate_limit() -> None:
    """One 429 waited out. Counted because it is what slow days are made of."""
    report = _current.get()
    if report is not None:
        report.rate_limited += 1
