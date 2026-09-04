import json
import re

from marshmallow import Schema, fields, pre_load, validate

class BookDetailsSchema(Schema):
    BookName = fields.Str(required=True)
    BookType = fields.Str(required=True)
    # Stored file name of the uploaded PDF, e.g. "1_734.pdf". Optional so a
    # caller that does not send it still validates; without it that book simply
    # cannot be retrieved from, and generation falls back to the title alone.
    FilePath = fields.Str(required=False, load_default="")

class RuleSchema(Schema):
    questionId = fields.Int(required=True)
    questionType = fields.Str(required=True)
    difficultyLevel = fields.Str(required=True)
    cognitiveLevel = fields.Str(required=True)
    mark = fields.Int(required=True)
    numberOfQuestions = fields.Int(required=True, validate=lambda n: n > 0)
    courseOutcome = fields.Str(required=True)

class GenerateSchema(Schema):
    module = fields.Str(required=True)
    unit = fields.Str(required=False, load_default="")
    content = fields.Str(required=True)
    Rules = fields.List(fields.Nested(RuleSchema), required=True)
    BookDetails = fields.List(fields.Nested(BookDetailsSchema), required=True)
    model = fields.Str(
        required=True,
        validate=validate.OneOf(["gemini", "openai", "deepseek", "claude"])
    )

# A backslash starting what can only be a LaTeX command: two or more letters
# after it. \times, \frac, \begin. Deliberately NOT \n or \t on their own,
# which stay as the JSON escapes they look like.
_LATEX_COMMAND = re.compile(r'\\(?=[A-Za-z]{2,})')

# Any remaining backslash that JSON does not recognise as an escape at all.
_UNKNOWN_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')

# What a backslash must become for json to read it as a literal backslash.
# Substituted through a FUNCTION, because re.sub interprets escapes in a
# replacement STRING and would collapse the pair straight back to one.
BACKSLASH_PAIR = chr(92) * 2


def _double_backslash(match):
    return BACKSLASH_PAIR



# A control character with a word running straight on from it: the fingerprint
# of a LaTeX command that json has already eaten. A tab followed by "imes" was
# \times; a form feed followed by "rac" was \frac. Ordinary text does not
# do this - a real tab is followed by a space, a number, or a line of a table.
_MANGLED_COMMAND = re.compile(r'[\t\x08\x0c\r][A-Za-z]{2,}')


def _looks_mangled(value):
    """True if any string in the parsed structure shows the fingerprint above."""
    if isinstance(value, str):
        return bool(_MANGLED_COMMAND.search(value))
    if isinstance(value, dict):
        return any(_looks_mangled(v) for v in value.values())
    if isinstance(value, list):
        return any(_looks_mangled(v) for v in value)
    return False


def _loads_latex_json(text):
    """
    Parses a JSON string that may contain raw LaTeX, or returns None.

    A model that serialises its own answer writes LaTeX straight into the
    string, so "answer_latex" arrives holding \times rather than \\times.
    That is wrong in two different ways, and the dangerous one is silent:
    \t and \f ARE valid JSON escapes, so the parse SUCCEEDS and quietly
    turns \times into a tab and \frac into a form feed. Only \alpha and
    its like fail loudly.

    So the parse is judged, not just attempted. Repair happens when the text
    will not parse at all, OR when what it parsed into carries the fingerprint
    of an eaten command. Payloads that are simply correct - including ones with
    a genuine \n newline, and ones where the LaTeX was properly escaped all
    along - are returned exactly as json read them, untouched.
    """
    if not isinstance(text, str):
        return None

    try:
        # strict=False tolerates literal newlines and tabs inside strings,
        # which hand-written JSON frequently contains.
        parsed = json.loads(text, strict=False)
    except (TypeError, ValueError):
        parsed = None

    if parsed is not None and not _looks_mangled(parsed):
        return parsed

    for pattern in (_LATEX_COMMAND, _UNKNOWN_ESCAPE):
        try:
            return json.loads(pattern.sub(_double_backslash, text), strict=False)
        except (TypeError, ValueError):
            continue

    # A mangled parse is still better than discarding a full set of questions.
    return parsed


class _LLMQuestionSchema(Schema):
    """
    INTERNAL: Validates the structure of a SINGLE question object
    as returned by the LLM tool call.
    """
    question = fields.Str(required=True)
    answer = fields.Str(required=True)
    question_latex = fields.Str(required=True)
    answer_latex = fields.Str(required=True)

class LLMToolOutputSchema(Schema):
    """
    PUBLIC: Validates the complete arguments payload we expect
    from the LLM's tool call.
    """
    questions = fields.List(
        fields.Nested(_LLMQuestionSchema()),
        required=True
    )

    @pre_load
    def _unwrap_stringified_questions(self, data, **kwargs):
        """
        Accepts a questions array that arrived as a JSON STRING.

        The tool schema asks for an array and the providers usually send one,
        but a model will sometimes serialise it and send
        {"questions": "[{...}]"} instead. That is a complete answer in the wrong
        wrapper, and rejecting it threw away a full set of questions the teacher
        had already waited half a minute for. Anything that does not parse into
        a list is left exactly as it came, so genuinely malformed output still
        fails validation rather than being smuggled through.
        """
        if not isinstance(data, dict) or not isinstance(data.get("questions"), str):
            return data

        unwrapped = _loads_latex_json(data["questions"])
        return {**data, "questions": unwrapped} if isinstance(unwrapped, list) else data