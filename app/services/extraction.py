"""
Filing questions a teacher already has, rather than writing new ones.

The reverse of generation. There, a teacher describes what they want and the
model invents questions. Here they upload a spreadsheet of questions they have
already written, and the model proposes where each one belongs in the syllabus
- its module, unit, type, difficulty, outcome, cognitive level and mark.

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

from app import usage
from app.config import Config

logger = logging.getLogger(__name__)

# Claude bills its reasoning against max_tokens alongside the answer.
_MAX_TOKENS = 16000


def _options(label: str, values) -> str:
    """One line of the vocabulary the model must choose from."""
    listed = ", ".join(str(v) for v in values if str(v).strip())
    return f"    {label}: {listed}\n" if listed else ""



async def _call_tool(model: str, instructions: str, message: str,
                     tool: dict, tool_choice: dict, key: str):
    """
    One model call that must answer by calling `tool`, whichever provider it is.

    The two SDKs differ in how a tool is declared, how the call is forced and
    where the answer is found - so the difference is confined here rather than
    repeated in every caller.
    """
    from app.extensions import async_clients

    if model == "gemini":
        # system_instruction, not part of contents: Gemini's implicit cache
        # matches on a shared prefix, and the instruction sits in front of
        # everything the caller varies.
        response = await async_clients.gemini.aio.models.generate_content(
            model=current_app.config["GEMINI_MODEL_NAME"],
            contents=message,
            config=_gemini_config(tool, instructions),
        )
        usage.record_llm("gemini", current_app.config["GEMINI_MODEL_NAME"], response)
        args = _gemini_tool_result(response, tool["name"])
        if args is None:
            logger.error("Gemini did not call %s", tool["name"])
            return []
        found = args.get(key)
    else:
        # cache_control on the system block caches the tool definition and
        # these instructions together - Claude caches tools, then system, then
        # messages, in that order, so one breakpoint here covers both.
        response = await async_clients.claude.messages.create(
            model=current_app.config["CLAUDE_MODEL_NAME"],
            max_tokens=_MAX_TOKENS,
            tools=[tool],
            tool_choice=tool_choice,
            system=[{
                "type": "text",
                "text": instructions,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": message}],
        )
        usage.record_llm("claude", current_app.config["CLAUDE_MODEL_NAME"], response)
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


def _gemini_config(tool: dict, instructions: str | None = None):
    """A GenerateContentConfig that forces one call to `tool`."""
    from google.genai import types

    return types.GenerateContentConfig(
        system_instruction=instructions,
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



async def _run_batches(calls: list) -> list:
    """
    Runs the first call alone, then the rest together.

    Not an accident of style: a cache entry only exists once a call that WROTE
    it has come back. Firing every batch at once - as this did - means none of
    them finds an entry, so each pays the 1.25x write premium for the same
    prefix and the cache never earns anything. One call first, then the fan-out,
    turns those writes into reads at a tenth of the price.

    The cost is one round-trip of latency. With a single batch there is nothing
    to warm, so nothing is paid.
    """
    if not calls:
        return []

    first = await asyncio.gather(calls[0], return_exceptions=True)
    if len(calls) == 1:
        return first

    rest = await asyncio.gather(*calls[1:], return_exceptions=True)
    return first + rest


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


def build_classify_instructions(vocabulary: dict) -> str:
    """
    The rules and the course's vocabulary - identical for every batch.

    Sent as a system prompt so it sits in front of the questions rather than
    after them, which is the only place a prompt cache can reach. See
    build_instructions for the reasoning; the same 1024-token prefix rule
    applies here.
    """
    return f"""You are filing a set of exam questions against a course syllabus.

The questions follow in the next message. For EVERY one of them, return an
entry carrying its number and where it belongs. Follow these rules exactly:

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


def build_questions_message(numbered) -> str:
    """The only part that differs between one batch and the next."""
    body = "\n\n".join(f"{i}. {text}" for i, text in numbered)
    return f"{body}\n\n--- END OF QUESTIONS ---"


async def _classify_batch(model, numbered, vocabulary):
    """One model call over one batch of questions."""
    return await _call_tool(
        model,
        build_classify_instructions(vocabulary),
        build_questions_message(numbered),
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

    results = await _run_batches(
        [_classify_batch(model, batch, vocabulary) for batch in batches])

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
