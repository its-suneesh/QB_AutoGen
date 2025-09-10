import logging
import json
import asyncio
from flask import current_app
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from openai import APIError

from .extensions import gemini_tool, OPENAI_COMPATIBLE_TOOL, async_clients


error_logger = logging.getLogger('error')


class ServiceError(Exception):
    """Custom exception for service layer errors."""
    def __init__(self, message, status_code=503):
        super().__init__(message)
        self.status_code = status_code

def generate_prompt(module, rule, num_questions, book_details, content):
    """
    Generates a balanced, optimized prompt for the LLM.
    """
    book_references = "\n".join([f"- {b['BookName']} (Type: {b['BookType']})" for b in book_details])

    prompt = f"""
    Task: Generate {num_questions} questions based ONLY on the provided content.
    You MUST call the `submit_questions` tool with a valid JSON array.
    Your entire response must be a single, complete, and valid JSON object for the tool call. Do not truncate your response.

    Content: "{content}"
    Module: {module}
    Book References:
    {book_references}

    Follow these rules for each question:
    1.  **Parameters**:
        - Question Type: {rule['questionType']}
        - Difficulty: {rule['difficultyLevel']}
        - Cognitive Level: {rule['cognitiveLevel']}
        - Marks: {rule['mark']}
    2.  **Output Structure**: Each object in the JSON array must have 4 string keys: "question", "answer", "question_latex", "answer_latex".
    3.  **MCQ Rule**: If Question Type is "Multiple Choice(MCQ)", the 'question' must have A, B, C, D options, and the 'answer' must be only the correct letter (e.g., "C").
    4.  **Answer Length**: For {rule['mark']} marks, provide a concise but complete answer. For marks > 5, be less descriptive.
    5.  **LaTeX**: Use LaTeX in '_latex' fields for math. If none, copy the plain text.
    """
    return prompt


async def _generate_single_rule(provider_instance, prompt_text, provider_name):
    """Makes a single async API call to the specified provider."""
    # --- Retry logic has been removed as requested ---
    if provider_name == 'gemini':
        response = await provider_instance.generate_content_async(prompt_text)
        part = response.candidates[0].content.parts[0]
        if hasattr(part, 'function_call') and part.function_call.name == "submit_questions":
            return part.function_call.args.get("questions", [])
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
            return args.get("questions", [])
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
            generate_prompt(data['module'], rule, rule.get("numberOfQuestions", 1), data['BookDetails'], data['content']),
            provider_name
        )
        for rule in data['Rules']
    ]

    current_app.logger.info(f"Dispatching {len(tasks)} tasks to '{provider_name}' concurrently.")

    all_generated_questions = []
    try:
        results_from_api = await asyncio.gather(*tasks, return_exceptions=True)

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
                    "questionType": rule["questionType"]
                }
                all_generated_questions.append(final_question)

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

    return all_generated_questions