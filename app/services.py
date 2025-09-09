# app/services.py

import logging
import json
from flask import current_app
import google.generativeai as genai
# --- MODIFIED: Import the renamed tool ---
from .extensions import clients, gemini_tool, OPENAI_COMPATIBLE_TOOL

error_logger = logging.getLogger('error')

class ServiceError(Exception):
    """Custom exception for service layer errors."""
    def __init__(self, message, status_code=503):
        super().__init__(message)
        self.status_code = status_code

def generate_prompt(module, rule, num_questions, book_details, content):
    """
    Creates a detailed, structured prompt for the Gemini model.
    (This function remains unchanged as it is provider-agnostic)
    """
    book_references = "\n".join([f"- {b['BookName']} (Type: {b['BookType']})" for b in book_details])
    question_type = rule['questionType']
    difficulty_level = rule['difficultyLevel']
    cognitive_level = rule['cognitiveLevel']
    mark = rule['mark']
    answer_length_instruction = ""
    if mark <= 5:
        answer_length_instruction = "Provide a complete and comprehensive answer."
    else:
        answer_length_instruction = "Provide only the main and important points in the answer, do not elaborate."

    mcq_instruction = ""
    if question_type.lower() == "multiple choice(MCQ)":
        mcq_instruction = """
        For Multiple Choice Questions (MCQ):
        - The 'question' field MUST include the question stem followed by exactly four options labeled A, B, C, and D...
        (rest of your prompt)
        """
    
    prompt = f"""
    Context:
    Module Name: {module}
    Syllabus Content: {content}
    Book References:
    {book_references}

    Task:
    Based ONLY on the provided Context... generate {num_questions} unique questions...
    
    (rest of your prompt)
    """
    return prompt

# --- REWRITTEN FUNCTION LOGIC ---

def generate_questions_from_prompt(data):
    """Handles logic of calling the selected LLM provider and processing the response."""
    
    module = data['module']
    rules = data['Rules']
    book_details = data['BookDetails']
    content = data['content']
    # This is now the PROVIDER name (e.g., "gemini", "openai") thanks to our schema
    provider_name = data['model'] 
    
    all_generated_questions = []

    for rule in rules:
        num_questions = rule.get("numberOfQuestions", 1)
        prompt_text = generate_prompt(module, rule, num_questions, book_details, content)
        questions_from_api = []

        current_app.logger.info(f"Processing rule with provider '{provider_name}' to generate {num_questions} question(s)...")

        try:
            if provider_name == 'gemini':
                # 1. Get the specific model name from config
                model_name = current_app.config['GEMINI_MODEL_NAME']
                gemini_model = genai.GenerativeModel(
                    model_name=model_name,
                    tools=[gemini_tool],
                    tool_config={"function_calling_config": "ANY"}
                )
                response = gemini_model.generate_content(prompt_text)
                part = response.candidates[0].content.parts[0]

                if hasattr(part, 'function_call') and part.function_call.name == "submit_questions":
                    questions_from_api = part.function_call.args.get("questions", [])
                else:
                    error_logger.warning(f"Gemini did not use the 'submit_questions' tool.")
                    if hasattr(part, 'text'):
                        error_logger.warning(f"Gemini Response Text: {part.text}")
            
            elif provider_name in ['deepseek', 'openai']:
                # This logic block handles both OpenAI and DeepSeek
                
                if provider_name == 'deepseek':
                    client = clients.deepseek
                    model_name = current_app.config['DEEPSEEK_MODEL_NAME']
                else: # provider_name == 'openai'
                    client = clients.openai
                    model_name = current_app.config['OPENAI_MODEL_NAME']

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt_text}],
                    tools=[OPENAI_COMPATIBLE_TOOL], # Use the compatible tool format
                    tool_choice="auto"
                )
                tool_call = response.choices[0].message.tool_calls[0]
                if tool_call.function.name == "submit_questions":
                    args = json.loads(tool_call.function.arguments)
                    questions_from_api = args.get("questions", [])
            
            else:
                # This case should theoretically not be hit thanks to our Schema validation,
                # but it's good practice to keep it as a fallback.
                raise ServiceError(f"Unsupported model provider: {provider_name}", status_code=400)

            # --- Response processing (this part remains the same) ---
            for q in questions_from_api:
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

        except (IndexError, AttributeError, KeyError, json.JSONDecodeError) as e:
            error_logger.error(f"Error processing model response from {provider_name}: {e}")
            if 'response' in locals() and response:
                 error_logger.error(f"Full Response: {response}")
            raise ServiceError(f"Failed to process the model's response. Please check logs.", status_code=500)

        except Exception as e:
            error_logger.error(f"An unexpected error occurred while calling the {provider_name} API: {e}", exc_info=True)
            raise ServiceError(f"An external service required for question generation is currently unavailable: {e}", status_code=503)
            
    return all_generated_questions