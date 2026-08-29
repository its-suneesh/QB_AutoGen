import logging
import json
import re
import asyncio
from flask import current_app
from google.genai import errors as genai_errors
from marshmallow import ValidationError
from openai import APIError as OpenAIAPIError

from .extensions import GEMINI_CONFIG, OPENAI_COMPATIBLE_TOOL, async_clients
from .schemas import LLMToolOutputSchema


error_logger = logging.getLogger('error')


class ServiceError(Exception):
    """Custom exception for service layer errors."""
    def __init__(self, message, status_code=503):
        super().__init__(message)
        self.status_code = status_code


# A command written with two backslashes (over-escaped), versus one written
# correctly with a single backslash. A genuine LaTeX "\\" line break is followed
# by whitespace, not by a letter or a brace, so it is not matched here.
_OVER_ESCAPED = re.compile(r'\\\\[A-Za-z{}]')
_ALREADY_CORRECT = re.compile(r'(?<!\\)\\[A-Za-z]')


def _unescape_llm_latex(value):
    """
    Repairs LaTeX the model has JSON-escaped one time too many.

    The model sometimes emits the JSON-escaped FORM of its LaTeX as the string's
    content, so "\\frac" reaches the client as two backslashes. In TeX "\\" is a
    line break, so the renderer reads it as a break followed by the literal word
    "frac" and the fraction never appears. Newlines arrive the same way, as the
    two characters backslash-n instead of a real line break.

    Measured over 50 generated questions, 22% came back like this - and it varied
    per response rather than per topic, which is why the fault looked intermittent.

    Decoding is all-or-nothing, so it is applied only when the string shows
    over-escaping AND contains no correctly escaped command. In the same sample
    the model's escaping was uniform within each string (11 wholly over-escaped,
    27 wholly correct, 0 mixed), so a whole-string decode is the right operation;
    the second condition enforces that rather than assuming it. A uniformly
    over-escaped string carries its row breaks as four backslashes, which decode
    back to the correct two.
    """
    if not value or not isinstance(value, str):
        return value
    if not _OVER_ESCAPED.search(value) or _ALREADY_CORRECT.search(value):
        return value
    try:
        # strict=False so real control characters inside the string are tolerated.
        return json.loads('"%s"' % value.replace('"', '\\"'), strict=False)
    except (ValueError, TypeError):
        return value

def _is_multiple_choice(question_type):
    """
    Decides whether a question type means "multiple choice".

    Deliberately NOT a fixed list of names. Question types are a per-institution
    master that users maintain themselves (QuestionTypeID / QuestionTypeName /
    ShortName), so the wording differs between institutions, new types get added,
    and the text carries real typos - "Diagram Qyestion" is live data. Any list
    hard-coded here would be wrong for the next institution that installs this.

    Matching is therefore on meaning, after stripping case and punctuation:
    anything containing "mcq", or both "multiple" and "choice", or "objective".
    "6 Mark Question", "Diagram Qyestion" and "Table Question" do not match.
    """
    normalised = re.sub(r"[^a-z0-9]+", " ", str(question_type or "").lower())
    if "mcq" in normalised.split():
        return True
    if "mcq" in normalised.replace(" ", ""):
        return True
    if "multiple" in normalised and "choice" in normalised:
        return True
    return "objective" in normalised


def generate_prompt(module, unit, rule, num_questions, book_details, content):
    book_references = "\n".join([f"- {b['BookName']} (Type: {b['BookType']})" for b in book_details])
    course_outcome = rule.get('courseOutcome', '')
    unit_line = f"Unit: {unit}" if unit else ""

    # The MCQ rule carries worked "\item[A)] first choice" examples. Left in the
    # prompt unconditionally it was shown to the model for EVERY question type,
    # including diagram and descriptive ones - so answer-choice lists kept
    # appearing in questions that should never have had them. The rule is only
    # sent when the question actually is multiple-choice; otherwise the model is
    # told plainly not to produce choices.
    question_type = str(rule.get("questionType", ""))
    is_mcq = _is_multiple_choice(question_type)

    if is_mcq:
        mcq_rule = r"""5.  **MCQ Rule**: This question IS multiple-choice.
        - Write the question first, then the four choices, as part of the question itself.
        - In "question": put each choice on its OWN line, labelled A) B) C) D).
        - In "question_latex": write the question, then the choices as a LIST so they do not run together on one line, using exactly this form:
          \begin{enumerate}
          \item[A)] first choice
          \item[B)] second choice
          \item[C)] third choice
          \item[D)] fourth choice
          \end{enumerate}
          Each choice still follows rule 7 - any mathematics inside it must be wrapped in $...$.
        - "answer" must be only the correct letter (e.g. "C"), matching the labels above."""
    else:
        mcq_rule = (
            '5.  **Not Multiple-Choice**: This question is "' + question_type + '". '
            "Do NOT write answer choices. There must be no A) B) C) D) options in the "
            '"question" or in "question_latex". Ask the question directly and let the '
            '"answer" be the full worked response.'
        )

    # RAW f-string. Without the r, Python interprets the backslash escapes in the
    # LaTeX examples below: "\frac" becomes a form-feed character followed by
    # "rac", and "\x" is outright a SyntaxError. The model was being shown a
    # control character where the rules meant to show it \frac.
    # Literal braces still have to be doubled - {{ }} - because this is an
    # f-string; single braces are replacement fields.
    prompt = rf"""
    Task: Generate exactly {num_questions} questions based ONLY on the provided content.

    RESPONSE CONTRACT (MANDATORY):
    - You MUST respond with exactly one function call to the `submit_questions` tool.
    - The `questions` argument MUST be a JSON array containing exactly {num_questions} question objects.
    - Your ENTIRE response must be that single function call: no text before or after it, no markdown, no commentary.
    - The JSON MUST be complete and valid. A partial, truncated, or malformed response will be rejected.

    Content: "{content}"
    Module: {module}
    {unit_line}
    Book References:
    {book_references}

    Follow these rules for each question:
    1.  **Parameters**:
        - Question Type: {rule['questionType']}
        - Difficulty: {rule['difficultyLevel']}
        - Cognitive Level: {rule['cognitiveLevel']}
        - Marks: {rule['mark']}
        - Course Outcome: {course_outcome}

    2.  **Cognitive Level Guide**: Align the question with the requested "{rule['cognitiveLevel']}" level using these examples as a structural guide:
        - Remembering: "State the formula for...", "List the components of...", "Identify the correct term..."
        - Understanding: "Explain the difference between...", "Summarize the process of...", "Describe how X works..."
        - Applying: "Calculate the value of X given...", "Use the theorem to solve...", "Write a function that..."
        - Analyzing: "Analyze the data flow in...", "Identify the flaw in the given code...", "Compare the efficiency of..."
        - Evaluating: "Assess the effectiveness of...", "Justify the use of Method A over Method B..."
        - Creating: "Design a system that...", "Formulate a new equation for...", "Build a comprehensive layout..."

    3.  **Course Outcome Alignment**: Ensure the question directly tests or maps to the stated Course Outcome: "{course_outcome}".

    4.  **Output Schema (MANDATORY)**: Each object in the `questions` array MUST be a valid JSON object containing ONLY these 4 keys, all with string values:
        - "question"
        - "answer"
        - "question_latex"
        - "answer_latex"
        Exactly {num_questions} objects must be returned. No missing keys and no extra keys.

    {mcq_rule}

    6.  **Answer Length**: For {rule['mark']} marks, provide a concise but complete answer. For marks > 5, be less descriptive.

    7.  **LaTeX Delimiters (CRITICAL)**: The "_latex" fields carry the mathematics.
        - EVERY mathematical expression MUST be wrapped in $...$ for inline maths, or $$...$$ for display maths.
        - A command such as \int, \frac, \sum, \lim or \sqrt, and every ^ or _, MUST sit inside those delimiters.
        - WRONG: Evaluate the integral \int_0^1 x^2 dx
          RIGHT: Evaluate the integral $\int_0^1 x^2 dx$
        - WRONG: the parabola y^2 = 2x meets the line
          RIGHT: the parabola $y^2 = 2x$ meets the line
        - Mathematics written without delimiters cannot be shown to the teacher and corrupts the printed question paper, so this rule is absolute.
        - If a question genuinely contains no mathematics, copy the plain text across unchanged.

    8.  **Backslashes and Newlines**: Write every LaTeX command with a SINGLE backslash - \frac, not \\frac. Do not escape backslashes. Use real line breaks, never the two characters \n.

    9.  **Plain Fields**: "question" and "answer" must be ordinary readable text containing NO LaTeX syntax at all - no backslash commands, no ^, no _, no $. Express the same mathematics there in words or plain notation.

    10. **Language Rule**: Use the language of the title of the reference book/textbook. (e.g., for a Malayalam book, the response should also be in Malayalam.)

    11. **Mathematics Rule**: If the reference book and module relate to Mathematics, the questions should be mathematical, i.e., more numerical problems rather than theoretical ones.

    12. **Diagrams**: When a question is genuinely clearer with a figure - coordinate geometry, the graph of a function, a triangle or circle construction, a tree, a network, a circuit, a chemical structure, a vector diagram - include ONE figure in "question_latex".
        - Give the PICTURE ONLY, in one of these forms: \begin{{tikzpicture}} ... \end{{tikzpicture}}, \begin{{circuitikz}} ... \end{{circuitikz}}, \begin{{pspicture}} ... \end{{pspicture}}, \begin{{forest}} ... \end{{forest}}, \begin{{venndiagram3sets}} ... \end{{venndiagram3sets}}, \begin{{tikzcd}} ... \end{{tikzcd}}, \begin{{modiagram}} ... \end{{modiagram}}, \begin{{asy}} ... \end{{asy}}, \chemfig{{...}}, \smartdiagram[...]{{...}}, or \feynmandiagram [...] {{...}}. Never write \documentclass, \usepackage, \usetikzlibrary or \begin{{document}} - the question paper supplies all of that, and a second \documentclass inside it breaks the whole paper rather than the one question.
        - It is a TEXT-mode environment: do NOT put it inside $...$. Rule 7 does not apply to the contents of a picture.
        - These packages are already loaded and may be used freely. Choose the one that fits the subject instead of drawing everything with plain tikz - the specialised package gets the conventions right (arrow styles, level spacing, ray tracing) where a hand-drawn tikz version usually does not:
            * Mathematics - pgfplots (\begin{{axis}}, \addplot) for the graph of a function or a data plot; tkz-euclide (\tkzDefPoint, \tkzDrawPolygon, \tkzDrawCircle, \tkzMarkAngle) for geometry constructions; tkz-graph (\SetGraphUnit, \Vertex, \Edge) for graph theory; venndiagram (venndiagram2sets, venndiagram3sets, \fillACapB and friends) for sets; tikz-cd for commutative diagrams; asymptote (\begin{{asy}} ... \end{{asy}}) when the figure needs loops, functions or real 3D.
            * Physics - circuitikz for circuits; pst-optic (\lens, \mirror inside \begin{{pspicture}}) for ray diagrams through lenses and mirrors; tikz-3dplot (\tdplotsetmaincoords, tdplot_main_coords) for 3D axes, vectors and solids; tikz-feynman (\feynmandiagram) for particle interactions; physics (\dv, \pdv, \grad, \curl, \abs, \norm, \ket); siunitx (\qty{{9.8}}{{\meter\per\second\squared}}, \si) for every quantity with a unit.
            * Chemistry - chemfig for structural formulas, rings and mechanisms; mhchem (\ce{{H2SO4 + 2NaOH -> Na2SO4 + 2H2O}}) for equations; chemformula (\ch) as the alternative to it; chemmacros (\ox{{2,Ca}} for oxidation numbers, \pH); modiagram for molecular orbital diagrams - in \molecule the keys are 1sMO, 2sMO and 2pMO, NOT 1s/2s/2p, which silently draw nothing.
            * Biology and general - forest for classification trees, cladograms and pedigree charts (it computes the spacing, unlike tikz's trees library); smartdiagram[circular diagram]{{...}} for life cycles and processes; amssymb.
        - graphicx and svg are NOT usable here: this figure arrives as code and there is no file to include. Everything must be drawn by the code itself.
        - These tikz libraries are already loaded and may be used freely, WITHOUT writing \usetikzlibrary: arrows.meta, calc, positioning, fit, matrix, chains, shapes.geometric, shapes.misc, shapes.symbols, patterns, patterns.meta, intersections, through, angles, quotes, decorations.markings, decorations.pathmorphing, decorations.pathreplacing, decorations.text, backgrounds, plotmarks, trees, 3d, fadings, calendar.
        - Prefer pgfplots for the graph of a function: it draws the axes, ticks and labels itself, which comes out more accurate than placing them by hand.
        - For a hand-plotted curve use the variable \x, for example: \draw[domain=-2:2] plot (\x, {{\x*\x}});
        - Write multiplication EXPLICITLY: \x*\x not \x\x, 2*\x not 2\x, 3*(\x+1) not 3(\x+1). The plotting parser rejects the implicit forms outright.
        - ANY non-English text inside a picture MUST be wrapped: \node {{\foreignlanguage{{malayalam}}{{സമയം}}}}, and likewise for tamil, hindi, arabic and the rest. This one is not cosmetic and not optional. Written bare, the label compiles with NO error, reports success, and draws NOTHING - the characters fall back to a Latin font that has no such glyph. The paper then prints with an unlabelled diagram and nobody finds out until it is in front of the students.
        - Keep the picture under about 25 lines and label axes or points with \node.
        - In the plain "question" field, describe the figure in words instead - the plain field must stay free of LaTeX (rule 9).
        - If a diagram adds nothing to the question, omit it. Never include a decorative figure.
    """
    return prompt


# HTTP statuses worth asking again for. All of them mean "not now" rather than
# "no": the provider is rate-limiting us or is briefly unwell, and the identical
# request usually succeeds moments later.
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

# Attempts INCLUDING the first. Three is what the 180s budget in
# generate_questions_from_prompt_async allows: a generation observed at up to
# ~45s, three of them plus the 1s and 2s waits is ~138s, still inside it. A
# fourth would risk the whole batch timing out to save one rule.
_MAX_ATTEMPTS = 3


def _is_retryable(error):
    """
    True only for a TEMPORARY provider failure.

    Retrying a permanent one is worse than not retrying at all: a blocked API key
    (403) or a wrong model name (404) answers identically every time, so the
    teacher waits three times as long for the same failure and three times the
    quota is spent. Both were seen in practice - API_KEY_SERVICE_BLOCKED and
    "models/... is not found" - which is why this is a strict allowlist rather
    than "retry anything that raised".
    """
    status = getattr(error, "code", None)
    if status is None:
        status = getattr(error, "status_code", None)
    if isinstance(status, int) and status in _RETRYABLE_STATUSES:
        return True
    # google-genai puts the HTTP status on every APIError, so the check above
    # already catches 429 and the 5xx list by code. ServerError - any 5xx - is
    # the backstop for a transient status that is not in it.
    return isinstance(error, genai_errors.ServerError)


async def _generate_single_rule(provider_instance, prompt_text, provider_name):
    """
    Makes one provider call for one rule, retrying a temporary failure.

    Retries here rather than in the caller so one rule backing off does not hold
    up the others - they are dispatched concurrently and each retries on its own.
    """
    last_error = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return await _call_provider(provider_instance, prompt_text, provider_name)
        except (genai_errors.APIError, OpenAIAPIError) as error:
            last_error = error
            if not _is_retryable(error) or attempt == _MAX_ATTEMPTS:
                raise
            # 1s, then 2s. Backing off rather than hammering gives a
            # rate-limited provider room to recover instead of adding to the
            # queue that caused the 429.
            delay = 2 ** (attempt - 1)
            error_logger.warning(
                f"{provider_name} returned a temporary error on attempt "
                f"{attempt}/{_MAX_ATTEMPTS} ({type(error).__name__}: {error}). "
                f"Retrying in {delay}s."
            )
            await asyncio.sleep(delay)

    # Unreachable: the loop either returns or raises.
    raise last_error


async def _call_provider(provider_instance, prompt_text, provider_name):
    """Makes a single async API call to the specified provider."""
    if provider_name == 'gemini':
        response = await provider_instance.aio.models.generate_content(
            model=current_app.config['GEMINI_MODEL_NAME'],
            contents=prompt_text,
            config=GEMINI_CONFIG,
        )
        # google-genai returns pydantic models, where an absent part is a field
        # holding None rather than a missing attribute - hasattr() is always
        # true here, so it has to be tested for None instead. parts itself is
        # None when the model returned no content at all.
        parts = response.candidates[0].content.parts or []
        part = parts[0] if parts else None
        if part is not None and part.function_call and part.function_call.name == "submit_questions":
            try:
                validated = LLMToolOutputSchema().load(dict(part.function_call.args))
                return validated["questions"]
            except ValidationError as e:
                error_logger.warning(
                    f"Gemini tool output failed schema validation: {e.messages}"
                )
                return []
        else:
            error_logger.warning(
                f"Gemini did not use the 'submit_questions' tool. "
                f"Response text: {getattr(part, 'text', 'N/A')}"
            )
            return []

    elif provider_name in ['deepseek', 'openai']:
        model_name = current_app.config['DEEPSEEK_MODEL_NAME'] if provider_name == 'deepseek' else current_app.config['OPENAI_MODEL_NAME']

        response = await provider_instance.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            tools=[OPENAI_COMPATIBLE_TOOL],
            tool_choice="auto"
        )
        tool_call = response.choices[0].message.tool_calls[0]
        if tool_call.function.name == "submit_questions":
            args = json.loads(tool_call.function.arguments)
            try:
                validated = LLMToolOutputSchema().load(args)
                return validated["questions"]
            except ValidationError as e:
                error_logger.warning(
                    f"{provider_name} tool output failed schema validation: {e.messages}"
                )
                return []
        return []

    raise ServiceError(f"Unsupported model provider: {provider_name}", status_code=400)


async def generate_questions_from_prompt_async(data):
    """
    Handles logic of calling the selected LLM provider concurrently for all rules.
    """
    provider_name = data['model']

    provider_instance = None
    if provider_name == 'gemini':
        # The model name and the tool ride with each request now (GEMINI_CONFIG
        # in _call_provider), so gemini hands back a reusable client here just
        # like the other two providers.
        provider_instance = async_clients.gemini
    elif provider_name == 'deepseek':
        provider_instance = async_clients.deepseek
    elif provider_name == 'openai':
        provider_instance = async_clients.openai

    if provider_instance is None:
        raise ServiceError(f"Unsupported model provider: {provider_name}", status_code=400)

    tasks = [
        _generate_single_rule(
            provider_instance,
            generate_prompt(data['module'], data.get('unit', ''), rule, rule.get("numberOfQuestions", 1), data['BookDetails'], data['content']),
            provider_name
        )
        for rule in data['Rules']
    ]

    current_app.logger.info(f"Dispatching {len(tasks)} tasks to '{provider_name}' concurrently.")

    all_generated_questions = []
    try:
        async with asyncio.timeout(180):
            results_from_api = await asyncio.gather(*tasks, return_exceptions=True)
    except TimeoutError:
        error_logger.error(
            f"Timed out after 180s waiting for {provider_name} responses for all rules."
        )
        raise ServiceError("The AI service took too long to respond.", status_code=504)
    except (genai_errors.APIError, OpenAIAPIError) as api_error:
        error_logger.error(
            f"API error during calls to {provider_name}: {api_error}", exc_info=True
        )
        raise ServiceError(f"An external service reported an error.", status_code=502)
    except Exception as e:
        error_logger.error(
            f"Unexpected error during API calls to {provider_name}: {e}", exc_info=True
        )
        raise ServiceError(f"An external service is unavailable.", status_code=503)

    failures = []

    for i, result in enumerate(results_from_api):
        rule = data['Rules'][i]
        if isinstance(result, Exception):
            error_logger.error(
                f"Error processing rule {rule['questionId']} with {provider_name}: {result}"
            )
            failures.append(result)
            continue

        for q in result:
            final_question = {
                "question": q.get("question", "").strip('"'),
                # Normalised here so every consumer receives usable LaTeX,
                # rather than each client having to repair it - see
                # _unescape_llm_latex.
                "questionLatex": _unescape_llm_latex(q.get("question_latex", "")).strip('"'),
                "answer": q.get("answer", "").strip('"'),
                "answerLatex": _unescape_llm_latex(q.get("answer_latex", "")).strip('"'),
                "cognitiveLevel": rule["cognitiveLevel"],
                "difficultyLevel": rule["difficultyLevel"],
                "mark": rule["mark"],
                "questionType": rule["questionType"],
                "courseOutcome": rule.get("courseOutcome", "") # <-- ADDED
            }
            all_generated_questions.append(final_question)

    # Nothing generated AND something failed is not a success.
    #
    # Returning 200 with an empty list here meant a blocked API key, a wrong
    # model name or an exhausted quota all reported themselves as "Successfully
    # generated 0 questions" - the reason stayed in the log, the caller saw a
    # normal response, and the teacher saw a spinner appear and leave. Both
    # cases were hit for real: API_KEY_SERVICE_BLOCKED, and a 404 for a model
    # name that did not exist.
    #
    # A PARTIAL failure still returns 200 with whatever succeeded: some
    # questions are more use to the teacher than an error, and the shortfall is
    # visible on screen.
    if failures and not all_generated_questions:
        error_logger.error(
            f"All {len(failures)} rule(s) failed against {provider_name}. "
            f"First error: {failures[0]}"
        )
        raise ServiceError(
            f"The AI provider rejected every request ({type(failures[0]).__name__}). "
            f"Check the API key, the model name and the remaining quota.",
            status_code=502,
        )

    if failures:
        current_app.logger.warning(
            f"{len(failures)} of {len(results_from_api)} rule(s) failed; "
            f"returning {len(all_generated_questions)} question(s) from the rest."
        )

    return all_generated_questions