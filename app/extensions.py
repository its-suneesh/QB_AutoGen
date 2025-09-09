from flask import current_app
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import google.generativeai as genai
from openai import OpenAI

jwt = JWTManager()
limiter = Limiter(
    key_func=get_remote_address,
)

gemini_tool = {
    "name": "submit_questions",
    "description": "Submits a list of generated questions.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "questions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "question": {"type": "STRING"},
                        "answer": {"type": "STRING"},
                        "question_latex": {"type": "STRING"},
                        "answer_latex": {"type": "STRING"}
                    },
                    "required": ["question", "answer", "question_latex", "answer_latex"]
                }
            }
        },
        "required": ["questions"]
    }
}

# --- RENAMED --- (from deepseek_tool)
# This format is compatible with both OpenAI and DeepSeek
OPENAI_COMPATIBLE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_questions",
        "description": "Submits a list of generated questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                            "question_latex": {"type": "string"},
                            "answer_latex": {"type": "string"}
                        },
                        "required": ["question", "answer", "question_latex", "answer_latex"]
                    }
                }
            },
            "required": ["questions"]
        }
    }
}

class ClientProvider:
    def __init__(self):
        self._deepseek_client = None
        self._openai_client = None 

    @property
    def deepseek(self):
        if self._deepseek_client is None:
            api_key = current_app.config.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY not set in config.")
            self._deepseek_client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )
        return self._deepseek_client
    
    @property
    def openai(self):
        if self._openai_client is None:
            api_key = current_app.config.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in config.")
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

clients = ClientProvider()