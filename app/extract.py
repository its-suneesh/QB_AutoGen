# app/extract.py
"""
Reading questions OUT of a PDF, rather than writing new ones.

The reverse of app/services.py. There, a teacher describes what they want and
the model invents questions. Here they hand over a paper that already has
questions in it - last year's exam, a question bank, a printed set - and the
model reads them out and proposes where each one belongs in the syllabus.

The difference is not cosmetic, and it changes what "good" means:

* Invention is a BUG here. A generated question can be phrased however the
  model likes; an extracted one must come back exactly as it was printed, or it
  is no longer the question that was set. Every returned question is therefore
  checked against the page it claims to come from, and one that does not appear
  there is flagged rather than trusted.

* The classification is a PROPOSAL, not an answer. Module, unit and course
  outcome are chosen from the lists this course actually has - never invented -
  and the model is told to leave them blank when the paper does not say. A
  blank is an empty dropdown the teacher fills in; a confident guess is a
  question filed under the wrong outcome for good, with nothing on screen to
  show it happened.

What comes back is shaped exactly like the rows the bulk-upload review grid
already edits, so the whole of that screen - the Module/Unit cascade, the LaTeX
preview, the validation, the commit - is reused unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from flask import current_app

from app.config import Config

logger = logging.getLogger(__name__)

# Pages per model call. A whole paper in one request risks running into
# max_tokens midway through the tool call, and a truncated tool call loses
# EVERY question in it, not just the last one. Smaller calls also run
# concurrently, so the batching costs no wall-clock time.
PAGES_PER_BATCH = 4

# Bounds one upload. A question paper is short; anything longer is a textbook
# sent to the wrong screen.
MAX_PAGES = 60

# Claude bills its reasoning against max_tokens alongside the answer.
_MAX_TOKENS = 16000

# Below this there is nothing to read and the file really is a scan. Well under
# a single short question, so a sparse paper is never mistaken for an image.
MIN_READABLE_CHARS = 40


EXTRACT_TOOL = {
    "name": "submit_extracted_questions",
    "description": (
        "Submits the questions found in the pages provided. Call this exactly "
        "once, with every question found and nothing else."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "Question": {
                            "type": "string",
                            "description":
                                "The question EXACTLY as printed, word for word. "
                                "Do not rephrase, correct, complete or shorten it.",
                        },
                        "AnswerKey": {
                            "type": "string",
                            "description":
                                "The answer if the document gives one, otherwise "
                                "an empty string. Do not write one yourself "
                                "unless explicitly asked to.",
                        },
                        "Mark": {
                            "type": ["number", "null"],
                            "description":
                                "Marks printed against the question, e.g. 5 from "
                                "'(5 marks)'. null if the paper does not say.",
                        },
                        "ModuleLabel": {"type": "string"},
                        "UnitLabel": {"type": "string"},
                        "QuestionTypeName": {"type": "string"},
                        "DifficultyLevelName": {"type": "string"},
                        "COCode": {"type": "string"},
                        "CognitiveLevelName": {"type": "string"},
                        "PageNo": {
                            "type": ["number", "null"],
                            "description": "Page the question was printed on.",
                        },
                    },
                    "required": ["Question"],
                },
            }
        },
        "required": ["questions"],
    },
}

EXTRACT_TOOL_CHOICE = {
    "type": "tool",
    "name": "submit_extracted_questions",
    "disable_parallel_tool_use": True,
}


def _options(label: str, values) -> str:
    """One line of the vocabulary the model must choose from."""
    listed = ", ".join(str(v) for v in values if str(v).strip())
    return f"    {label}: {listed}\n" if listed else ""


def build_prompt(pages: list[tuple[int, str]], vocabulary: dict, want_answers: bool) -> str:
    """
    The instruction for one batch of pages.

    The vocabulary is spelled out so the model picks from what this course
    actually has. Asked for a "module" with no list, a model will happily
    return "Module 1" for a course whose modules are named after topics.
    """
    body = "\n\n".join(f"--- PAGE {no} ---\n{text}" for no, text in pages)

    answer_rule = (
        "Write a correct, complete answer for each question."
        if want_answers else
        "Leave \"AnswerKey\" empty unless the document itself prints the answer. "
        "Do NOT write answers of your own."
    )

    return f"""You are reading a printed question paper and listing the questions in it.

{body}

--- END OF PAGES ---

Copy out every question that appears above. Follow these rules exactly:

1.  VERBATIM. "Question" must be the question word for word as printed.
    Do not rephrase it, fix its grammar, expand abbreviations, or complete a
    sentence that looks unfinished. It is checked against the page text, and a
    question that has been reworded is rejected.

2.  A question only. The text must START at the first word of the question
    and END at its last. Strip the numbering ("1.", "Q3", "a)"), the marks
    ("(5 marks)", "(2 marks)"), and any trailing dots or spacing used to line
    the marks up - all of which belong in their own fields or nowhere. The
    wording in between stays untouched.

3.  Sub-questions. "12 a) ... (6 marks) b) ... (4 marks)" is TWO questions,
    each with its OWN marks - never one question carrying the total. Split
    whenever the parts have separate marks or can be answered separately, and
    keep them together only when part (b) cannot be understood without (a).
    Never return the "a)" and "b)" labels or their marks inside the text.

4.  Not questions. Ignore headings, instructions ("Answer any five"), the
    institution name, time and total marks, page numbers, and running headers.

5.  Mathematics and diagrams. Write mathematics as LaTeX between $...$. If a
    question refers to a figure that is drawn rather than written, keep the
    words and add [FIGURE] where the drawing was - do not attempt to describe
    or redraw it.

6.  {answer_rule}

7.  CLASSIFY BY CHOOSING FROM THESE LISTS, NEVER BY INVENTING. Return one of
    the listed values EXACTLY as written, or an empty string.

    Judge from the CONTENT of the question, which is usually enough:
      - a question about binary numbers belongs to the number-systems module,
        whatever the paper does or does not say;
      - "Define", "List", "State" is Knowledge and Easy; "Explain", "Describe"
        is Understanding; "Simplify", "Convert", "Calculate" is Application;
      - two marks is a short answer, ten marks is an essay.
    Say what the question is about. Do not leave a field blank merely because
    the paper does not print the word.

    Leave it empty ONLY when the content genuinely does not decide it - a
    course outcome, most often, since a paper rarely states one. An empty
    field is a blank box the teacher fills in; a value that contradicts the
    question is filed silently and never noticed.

{_options("ModuleLabel", vocabulary.get("modules", []))}{_options("UnitLabel", vocabulary.get("units", []))}{_options("QuestionTypeName", vocabulary.get("question_types", []))}{_options("DifficultyLevelName", vocabulary.get("difficulty_levels", []))}{_options("CognitiveLevelName", vocabulary.get("cognitive_levels", []))}{_options("COCode", vocabulary.get("course_outcomes", []))}
8.  Marks. Take them from the paper when printed. Do not invent a mark.

Return every question by calling submit_extracted_questions once."""


# --------------------------------------------------------------------------
# Did the model actually copy, or did it write?
# --------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercased, punctuation-free, single-spaced - for comparing wording."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _shingles(text: str, size: int) -> set:
    words = _normalise(text).split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


def verbatim_score(question: str, source: str) -> float:
    """
    How much of the question is actually present in the page text.

    Word runs rather than whole-string containment, because extraction
    legitimately drops the numbering, the marks and the line breaks. A
    reworded question shares few runs; a faithfully copied one shares nearly
    all of them.

    BOTH sides are cut at the same length, and that length comes from the
    question. Fixing it at six made every short question look invented:
    "Define an algorithm" is three words, so it could never match a run of six
    however faithfully it had been copied.
    """
    words = _normalise(question).split()
    if not words:
        return 0.0

    size = min(6, len(words))
    wanted = _shingles(question, size)
    if not wanted:
        return 0.0
    return len(wanted & _shingles(source, size)) / len(wanted)


# Below this, the wording is the model's rather than the paper's.
VERBATIM_THRESHOLD = 0.6


async def _call_tool(model: str, prompt: str, tool: dict, tool_choice: dict, key: str):
    """
    One model call that must answer by calling `tool`, whichever provider it is.

    The two SDKs differ in how a tool is declared, how the call is forced and
    where the answer is found - so the difference is confined here rather than
    repeated in every caller.
    """
    from app.extensions import async_clients

    if model == "gemini":
        response = await async_clients.gemini.aio.models.generate_content(
            model=current_app.config["GEMINI_MODEL_NAME"],
            contents=prompt,
            config=_gemini_config(tool),
        )
        args = _gemini_tool_result(response, tool["name"])
        if args is None:
            logger.error("Gemini did not call %s", tool["name"])
            return []
        found = args.get(key)
    else:
        response = await async_clients.claude.messages.create(
            model=current_app.config["CLAUDE_MODEL_NAME"],
            max_tokens=_MAX_TOKENS,
            tools=[tool],
            tool_choice=tool_choice,
            messages=[{"role": "user", "content": prompt}],
        )
        block = next(
            (b for b in response.content
             if b.type == "tool_use" and b.name == tool["name"]),
            None,
        )
        if block is None:
            logger.error("Claude did not call %s (stop_reason=%s)",
                         tool["name"], response.stop_reason)
            return []
        found = block.input.get(key)

    if isinstance(found, str):
        # The stringified-array quirk the generator hits too; reuse the repair
        # that already knows how to read LaTeX out of it.
        from app.schemas import _loads_latex_json
        found = _loads_latex_json(found)

    return found if isinstance(found, list) else []


async def _extract_batch(model, pages, vocabulary, want_answers):
    """One model call over one batch of pages."""
    return await _call_tool(
        model,
        build_prompt(pages, vocabulary, want_answers),
        EXTRACT_TOOL, EXTRACT_TOOL_CHOICE, "questions",
    )


# Models this can run on. Extraction uses a tool call whose shape differs per
# SDK, so a model is listed here only once it has actually been wired -
# accepting a name and quietly running something else would be worse than
# refusing it.
SUPPORTED_MODELS = ("claude", "gemini")


# Gemini's function-calling schema is a restricted OpenAPI subset: types are
# UPPERCASE and a type union is not allowed. Rather than keep a second copy of
# every tool in step by hand, the one schema above is translated.
_GEMINI_TYPES = {
    "object": "OBJECT", "array": "ARRAY", "string": "STRING",
    "number": "NUMBER", "integer": "INTEGER", "boolean": "BOOLEAN",
}


def _to_gemini_schema(node):
    """Rewrites a JSON-Schema fragment into the form Gemini accepts."""
    if not isinstance(node, dict):
        return node

    out = {}
    kind = node.get("type")
    if isinstance(kind, list):
        # ["number", "null"] -> NUMBER. Gemini has no union and no null type,
        # so nullability is carried by the field simply being absent - which is
        # why Mark and PageNo are not in any "required" list.
        kind = next((k for k in kind if k != "null"), "string")
    if kind:
        out["type"] = _GEMINI_TYPES.get(kind, kind.upper())

    if "description" in node:
        out["description"] = node["description"]
    if "properties" in node:
        out["properties"] = {k: _to_gemini_schema(v)
                             for k, v in node["properties"].items()}
    if "items" in node:
        out["items"] = _to_gemini_schema(node["items"])
    if "required" in node:
        out["required"] = node["required"]
    return out


def _gemini_config(tool: dict):
    """A GenerateContentConfig that forces one call to `tool`."""
    from google.genai import types

    return types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[{
            "name": tool["name"],
            "description": tool["description"],
            "parameters": _to_gemini_schema(tool["input_schema"]),
        }])],
        # What tool_choice does for Claude: answer by calling the tool, never
        # with prose.
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY")
        ),
    )


def _gemini_tool_result(response, tool_name: str):
    """The arguments of the tool call, or None if the model did not make one."""
    candidates = response.candidates or []
    if not candidates:
        return None

    # google-genai returns pydantic models, so an absent part is a field
    # holding None rather than a missing attribute - hasattr is always true.
    for part in (candidates[0].content.parts or []):
        call = getattr(part, "function_call", None)
        if call is not None and call.name == tool_name:
            return dict(call.args or {})
    return None


async def extract_from_pdf(pdf_bytes: bytes, vocabulary: dict,
                           model: str = "claude",
                           want_answers: bool = False) -> dict:
    """
    Reads every question out of a PDF and proposes where each one belongs.

    Returns rows shaped like the bulk-upload review grid, so the portal can
    show them in the screen that already exists for editing and committing
    questions in bulk.
    """
    from app import rag                      # extraction and quality, shared

    pages_text = rag.extract_pages(pdf_bytes)          # raises PdfUnreadable
    status, quality = rag.assess(pages_text)

    # rag.assess answers a different question - "is there enough text here to
    # be worth indexing as a textbook", which wants hundreds of characters a
    # page. A question paper is sparse by nature: a page of two-mark questions
    # can be a few hundred characters in total, and a one-line document is
    # perfectly readable while failing that bar completely. Judging extraction
    # by it told teachers to OCR files that were never scanned.
    #
    # What matters here is only whether there is ANY text to read.
    readable = sum(len(t.strip()) for t in pages_text)

    if readable < MIN_READABLE_CHARS:
        # A file with SOME text is not a scan, however little it has, and
        # telling its owner to OCR it sends them to fix something that is not
        # broken. The two cases are named apart.
        message = rag.STATUS_MESSAGES["needs_ocr"] if readable == 0 else (
            f"Only {readable} characters could be read from this PDF, which is "
            "too little to hold a question. If it is a scan, it needs OCR first."
        )
        return {
            "status": "needs_ocr",
            "message": message,
            "pages": len(pages_text),
            "quality": round(quality, 3),
            "questions": [],
        }

    # Text came out, but it is not letters - a broken font encoding rather than
    # a scan. Extraction from it would be nonsense that looks like questions.
    if status == "low_quality" and readable >= MIN_READABLE_CHARS * 4:
        return {
            "status": "low_quality",
            "message": rag.STATUS_MESSAGES["low_quality"],
            "pages": len(pages_text),
            "quality": round(quality, 3),
            "questions": [],
        }

    pages = [(n, t) for n, t in enumerate(pages_text, start=1) if t.strip()]
    truncated = len(pages) > MAX_PAGES
    pages = pages[:MAX_PAGES]

    batches = [pages[i:i + PAGES_PER_BATCH]
               for i in range(0, len(pages), PAGES_PER_BATCH)]

    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"'{model}' cannot read a question paper yet. Supported: "
            + ", ".join(SUPPORTED_MODELS)
        )

    results = await asyncio.gather(
        *(_extract_batch(model, batch, vocabulary, want_answers)
          for batch in batches),
        return_exceptions=True,
    )

    by_page = {no: text for no, text in pages}
    rows: list[dict] = []

    for batch, result in zip(batches, results):
        if isinstance(result, Exception):
            # One batch failing must not lose the rest of the paper - the
            # teacher keeps the questions that were read and can re-run for
            # the pages that were not.
            logger.exception("Extraction failed for pages %s: %s",
                             [n for n, _ in batch], result)
            continue

        source = "\n".join(text for _, text in batch)
        for item in result:
            row = _to_row(item, source, by_page)
            if row:
                rows.append(row)

    return {
        "status": "extracted",
        "message": _summarise(rows, truncated),
        "pages": len(pages_text),
        "quality": round(quality, 3),
        "truncated": truncated,
        "questions": rows,
    }


def _to_row(item: dict, source: str, by_page: dict) -> dict | None:
    """One model answer turned into a review-grid row, or dropped."""
    question = (item.get("Question") or "").strip()
    if len(question) < 10:
        return None                     # a fragment, not a question

    # Prefer the page the model named; fall back to the whole batch, because a
    # wrong page number should not fail an otherwise faithful question.
    page_no = item.get("PageNo")
    page_text = by_page.get(int(page_no)) if isinstance(page_no, (int, float)) else None
    score = max(verbatim_score(question, page_text or ""),
                verbatim_score(question, source))

    mark = item.get("Mark")
    return {
        "Question": question,
        "AnswerKey": (item.get("AnswerKey") or "").strip(),
        "Mark": int(mark) if isinstance(mark, (int, float)) else None,
        "ModuleLabel": (item.get("ModuleLabel") or "").strip(),
        "UnitLabel": (item.get("UnitLabel") or "").strip(),
        "QuestionTypeName": (item.get("QuestionTypeName") or "").strip(),
        "DifficultyLevelName": (item.get("DifficultyLevelName") or "").strip(),
        "COCode": (item.get("COCode") or "").strip(),
        "CognitiveLevelName": (item.get("CognitiveLevelName") or "").strip(),
        "PageNo": int(page_no) if isinstance(page_no, (int, float)) else None,
        # Surfaced in the grid so a reworded question is visible before it is
        # committed, instead of being discovered on a printed paper.
        "Verbatim": score >= VERBATIM_THRESHOLD,
        "VerbatimScore": round(score, 2),
    }


def _summarise(rows: list[dict], truncated: bool) -> str:
    if not rows:
        return ("No questions were found. If this is a question paper, its text "
                "may be too irregular to read.")

    reworded = sum(1 for r in rows if not r["Verbatim"])
    # Unit is deliberately not counted: it is optional, and a paper states a
    # topic rather than a unit number. Counting it made every question look
    # incomplete on courses that do not define units.
    unclassified = sum(
        1 for r in rows
        if not r["ModuleLabel"] or not r["COCode"] or not r["CognitiveLevelName"]
    )

    parts = [f"{len(rows)} question(s) found."]
    if reworded:
        parts.append(f"{reworded} may have been reworded - check the highlighted rows.")
    if unclassified:
        parts.append(f"{unclassified} need a module, outcome or cognitive level chosen.")
    if truncated:
        parts.append(f"Only the first {MAX_PAGES} pages were read.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# Classifying questions the teacher already has
# --------------------------------------------------------------------------
#
# The other half of this module. Extraction reads questions OUT of a paper and
# must not change a word of them; classification is handed the words already
# and only has to say where each belongs.
#
# That difference removes the hardest problem: there is no verbatim check here
# and no risk of invention, because the text never leaves the teacher's file.
# The model is asked for labels and nothing else.

# Questions per call. Larger than the page batch because a question is short
# and only its classification comes back - but still bounded, so one truncated
# reply cannot lose a whole spreadsheet.
QUESTIONS_PER_BATCH = 15


CLASSIFY_TOOL = {
    "name": "submit_classifications",
    "description": (
        "Submits one classification for every question given. Call this exactly "
        "once, with an entry for each numbered question and nothing else."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "Index": {
                            "type": "number",
                            "description": "The number of the question being classified.",
                        },
                        "ModuleLabel": {"type": "string"},
                        "UnitLabel": {"type": "string"},
                        "QuestionTypeName": {"type": "string"},
                        "DifficultyLevelName": {"type": "string"},
                        "COCode": {"type": "string"},
                        "CognitiveLevelName": {"type": "string"},
                        "Mark": {
                            "type": ["number", "null"],
                            "description":
                                "Marks the question is worth, if the wording implies "
                                "it. null when it does not.",
                        },
                    },
                    "required": ["Index"],
                },
            }
        },
        "required": ["classifications"],
    },
}

CLASSIFY_TOOL_CHOICE = {
    "type": "tool",
    "name": "submit_classifications",
    "disable_parallel_tool_use": True,
}


def build_classify_prompt(numbered: list[tuple[int, str]], vocabulary: dict) -> str:
    """The instruction for one batch of questions."""
    body = "\n\n".join(f"{i}. {text}" for i, text in numbered)

    return f"""You are filing a set of exam questions against a course syllabus.

{body}

--- END OF QUESTIONS ---

For EVERY question above, return one entry carrying its number and where it
belongs. Follow these rules exactly:

1.  Do not rewrite, answer, correct or comment on any question. Only classify.

2.  CHOOSE FROM THESE LISTS, NEVER INVENT. Return one of the listed values
    EXACTLY as written, or an empty string.

    Judge from the CONTENT of the question:
      - a question about binary numbers belongs to the number-systems module,
        whatever else it mentions;
      - "Define", "List", "State" is Knowledge and Easy; "Explain", "Describe"
        is Understanding; "Simplify", "Convert", "Calculate", "Solve" is
        Application; "Compare", "Analyse", "Justify" is Analysis.

    Leave a field empty ONLY when the question genuinely does not decide it.
    An empty field is a blank box the teacher fills in; a value that
    contradicts the question is filed silently and never noticed.

{_options("ModuleLabel", vocabulary.get("modules", []))}{_options("UnitLabel", vocabulary.get("units", []))}{_options("QuestionTypeName", vocabulary.get("question_types", []))}{_options("DifficultyLevelName", vocabulary.get("difficulty_levels", []))}{_options("CognitiveLevelName", vocabulary.get("cognitive_levels", []))}{_options("COCode", vocabulary.get("course_outcomes", []))}
3.  Marks. Give the marks the question is plainly worth from its scope - a
    one-line definition is small, a "discuss with examples" is large. Use null
    if the wording does not suggest one.

4.  Return an entry for EVERY number above, even where every field is empty.

Answer by calling submit_classifications once."""


async def _classify_batch(model, numbered, vocabulary):
    """One model call over one batch of questions."""
    return await _call_tool(
        model,
        build_classify_prompt(numbered, vocabulary),
        CLASSIFY_TOOL, CLASSIFY_TOOL_CHOICE, "classifications",
    )


async def classify_questions(questions: list[str], vocabulary: dict,
                             model: str = "claude") -> dict:
    """
    Files a list of questions the teacher already has against their syllabus.

    Returns rows in the same shape extraction returns, so the review screen and
    the save path do not care which of the two produced them.

    The question text is passed through UNCHANGED - it came from the teacher's
    own file and the model is only asked for labels, so there is nothing here
    to verify or to get wrong about the wording.
    """
    cleaned = [(i, q.strip()) for i, q in enumerate(questions or [], start=1)
               if q and q.strip()]

    if not cleaned:
        return {"status": "classified", "message": "No questions were given.",
                "questions": []}

    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"'{model}' cannot classify questions yet. Supported: "
            + ", ".join(SUPPORTED_MODELS)
        )

    batches = [cleaned[i:i + QUESTIONS_PER_BATCH]
               for i in range(0, len(cleaned), QUESTIONS_PER_BATCH)]

    results = await asyncio.gather(
        *(_classify_batch(model, batch, vocabulary)
          for batch in batches),
        return_exceptions=True,
    )

    # Keyed by the number the model was given, so a batch that comes back out
    # of order - or short - still lands against the right questions.
    by_index: dict[int, dict] = {}
    failed = 0
    for batch, result in zip(batches, results):
        if isinstance(result, Exception):
            # One batch failing must not lose the rest of the file. Those
            # questions are still returned, unclassified, for the teacher to
            # fill in rather than to upload again.
            logger.exception("Classification failed for %s questions: %s",
                             len(batch), result)
            failed += len(batch)
            continue
        for item in result:
            index = item.get("Index")
            if isinstance(index, (int, float)):
                by_index[int(index)] = item

    rows = []
    for index, text in cleaned:
        found = by_index.get(index, {})
        mark = found.get("Mark")
        rows.append({
            "Question": text,
            "AnswerKey": "",
            "Mark": int(mark) if isinstance(mark, (int, float)) else None,
            "ModuleLabel": (found.get("ModuleLabel") or "").strip(),
            "UnitLabel": (found.get("UnitLabel") or "").strip(),
            "QuestionTypeName": (found.get("QuestionTypeName") or "").strip(),
            "DifficultyLevelName": (found.get("DifficultyLevelName") or "").strip(),
            "COCode": (found.get("COCode") or "").strip(),
            "CognitiveLevelName": (found.get("CognitiveLevelName") or "").strip(),
        })

    unclassified = sum(1 for r in rows if not r["ModuleLabel"])
    parts = [f"{len(rows)} question(s) read from the file."]
    if failed:
        parts.append(f"{failed} could not be classified - fill those in yourself.")
    elif unclassified:
        parts.append(f"{unclassified} need a module chosen.")

    return {"status": "classified", "message": " ".join(parts), "questions": rows}
