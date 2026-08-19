import logging
import json
import asyncio
from flask import current_app
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from marshmallow import ValidationError
from openai import APIError

from .extensions import gemini_tool, OPENAI_COMPATIBLE_TOOL, async_clients
from .schemas import LLMToolOutputSchema


error_logger = logging.getLogger('error')


class ServiceError(Exception):
    """Custom exception for service layer errors."""
    def __init__(self, message, status_code=503):
        super().__init__(message)
        self.status_code = status_code

def generate_prompt(module, unit, rule, num_questions, book_details, content):
    book_references = "\n".join([f"- {b['BookName']} (Type: {b['BookType']})" for b in book_details])
    course_outcome = rule.get('courseOutcome', '')
    unit_line = f"Unit: {unit}" if unit else ""

    prompt = f"""
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

    5.  **MCQ Rule**: If Question Type is "Multiple Choice(MCQ)", the "question" must contain options A, B, C, D, and the "answer" must be only the correct letter (e.g., "C").

    6.  **Answer Length**: For {rule['mark']} marks, provide a concise but complete answer. For marks > 5, be less descriptive.

    7.  **LaTeX**: Use LaTeX in the "_latex" fields for mathematical content. If none is needed, copy the plain text.

    8.  **Language Rule**: Use the language of the title of the reference book/textbook. (e.g., for a Malayalam book, the response should also be in Malayalam.)

    9.  **Mathematics Rule**: If the reference book and module relate to Mathematics, the questions should be mathematical, i.e., more numerical problems rather than theoretical ones.
    """
    return prompt


async def _generate_single_rule(provider_instance, prompt_text, provider_name):
    """Makes a single async API call to the specified provider."""
    # --- Retry logic has been removed as requested ---
    if provider_name == 'gemini':
        response = await provider_instance.generate_content_async(prompt_text)
        part = response.candidates[0].content.parts[0]
        if hasattr(part, 'function_call') and part.function_call.name == "submit_questions":
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
        provider_instance = genai.GenerativeModel(
            model_name=current_app.config['GEMINI_MODEL_NAME'],
            tools=[gemini_tool],
            tool_config={"function_calling_config": "ANY"}
        )
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
    except (google_exceptions.GoogleAPICallError, APIError) as api_error:
        error_logger.error(
            f"API error during calls to {provider_name}: {api_error}", exc_info=True
        )
        raise ServiceError(f"An external service reported an error.", status_code=502)
    except Exception as e:
        error_logger.error(
            f"Unexpected error during API calls to {provider_name}: {e}", exc_info=True
        )
        raise ServiceError(f"An external service is unavailable.", status_code=503)

    for i, result in enumerate(results_from_api):
        rule = data['Rules'][i]
        if isinstance(result, Exception):
            error_logger.error(
                f"Error processing rule {rule['questionId']} with {provider_name}: {result}"
            )
            continue

        for q in result:
            final_question = {
                "question": q.get("question", "").strip('"'),
                "questionLatex": q.get("question_latex", "").strip('"'),
                "answer": q.get("answer", "").strip('"'),
                "answerLatex": q.get("answer_latex", "").strip('"'),
                "cognitiveLevel": rule["cognitiveLevel"],
                "difficultyLevel": rule["difficultyLevel"],
                "mark": rule["mark"],
                "questionType": rule["questionType"],
                "courseOutcome": rule.get("courseOutcome", "") # <-- ADDED
            }
            all_generated_questions.append(final_question)

    return all_generated_questions