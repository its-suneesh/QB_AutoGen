# app/services.py

import logging
import json
from flask import current_app
import google.generativeai as genai
from .extensions import clients, gemini_tool, deepseek_tool

error_logger = logging.getLogger('error')

def generate_prompt(module, rule, num_questions, book_details, content):
    """
   Creates a detailed, structured prompt for the Gemini model.
    This version asks for a specific number of unique questions with enhanced MCQ instructions.
    """
    book_references = "\n".join([f"- {b['BookName']} (Type: {b['BookType']})" for b in book_details])

    # Extract rule details for clarity in the prompt
    question_type = rule['questionType']
    difficulty_level = rule['difficultyLevel']
    cognitive_level = rule['cognitiveLevel']
    mark = rule['mark']

    # --- UPDATED LINE ---
    # Conditional instruction for answer length based on marks
    answer_length_instruction = ""
    if mark <= 5:
        answer_length_instruction = "Provide a complete and comprehensive answer."
    else:
        answer_length_instruction = "Provide only the main and important points in the answer, do not elaborate."


    # Enhanced instruction for MCQ type questions
    mcq_instruction = ""
    if question_type.lower() == "multiple choice(MCQ)":
        mcq_instruction = """
        For Multiple Choice Questions (MCQ):
        - The 'question' field MUST include the question stem followed by exactly four options labeled A, B, C, and D, each on a new line (e.g., 'What is 2+2?\nA. 1\nB. 2\nC. 4\nD. 5').
        - The 'answer' field MUST specify the correct option by its label and content (e.g., 'C. 4'), without any additional explanation unless marks ≤ 5.
        - The 'question_latex' field MUST include the question stem followed by the four options formatted in LaTeX using an itemize environment (e.g., '\\begin{itemize}\\item[A.] 1\\item[B.] 2\\item[C.] 4\\item[D.] 5\\end{itemize}').
        - The 'answer_latex' field MUST specify the correct option in LaTeX (e.g., 'C. 4'), without any additional explanation unless marks ≤ 5.
        - All four options MUST be distinct, plausible, and relevant to the question, with only one correct answer.
        - Ensure the question stem is clear and directly related to the provided syllabus content and book references.
        """
    
    prompt = f"""
    Context:
    Module Name: {module}
    Syllabus Content: {content}
    Book References:
    {book_references}

    Task:
    Based ONLY on the provided Context (Module Name, Syllabus Content,Book References and Question Generation Rules), generate {num_questions} unique questions and their corresponding answers that strictly adhere to the following criteria.
    Submit all of them together in a single list using the 'submit_questions' tool.

    Question Criteria:
    - Question Type: {question_type}
    - Difficulty Level: {difficulty_level}
    - Cognitive Level: {cognitive_level}
    - Mark: {mark}

    Answer Criteria:
    - {answer_length_instruction}

    *CRITICAL FORMATTING RULES:*
    1.  The 'question' and 'answer' fields must be plain text only. They MUST NOT contain any LaTeX commands or backslashes.
    2.  The 'question_latex' and 'answer_latex' fields should contain ONLY the LaTeX code snippet. DO NOT include \\documentclass, \\begin{{document}}, or \\end{{document}}.
    3.  For LaTeX tables, NEVER wrap the \\begin{{tabular}} or \\begin{{array}} environments in dollar signs ($).
    4.  Do not wrap any of the final strings in extra quotation marks.
    5.  {mcq_instruction}
    """
    return prompt

def generate_questions_from_prompt(data):
    """Handles the logic of calling the selected LLM API and processing the response."""
    
    module = data['module']
    rules = data['Rules']
    book_details = data['BookDetails']
    content = data['content']
    # --- Get the specific model name from the request ---
    model_name = data['model']
    
    all_generated_questions = []

    for rule in rules:
        num_questions = rule.get("numberOfQuestions", 1)
        current_app.logger.info(f"Processing rule with model '{model_name}' to generate {num_questions} question(s)...")

        prompt_text = generate_prompt(module, rule, num_questions, book_details, content)
        questions_from_api = []

        try:
            # --- Determine provider based on model name prefix ---
            if model_name.startswith('gemini'):
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
            
            elif model_name.startswith('deepseek'):
                client = clients.deepseek
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt_text}],
                    tools=[deepseek_tool],
                    tool_choice="auto"
                )
                tool_call = response.choices[0].message.tool_calls[0]
                if tool_call.function.name == "submit_questions":
                    args = json.loads(tool_call.function.arguments)
                    questions_from_api = args.get("questions", [])
            
            else:
                raise ValueError(f"Unsupported model provider for model name: {model_name}")

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
            error_logger.error(f"Error processing model response: {e}")
            if 'response' in locals() and response:
                 error_logger.error(f"Full Response: {response}")
        except Exception as e:
            error_logger.error(f"An unexpected error occurred while calling the API: {e}", exc_info=True)
            
    return all_generated_questions