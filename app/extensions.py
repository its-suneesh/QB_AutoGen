from flask import current_app
from google import genai
from google.genai import types
from openai import AsyncOpenAI

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

# What every Gemini call sends alongside the prompt.
#
# google-genai has no model object to hang these on - genai.Client is bound to
# the API key alone - so the model name, the tool and the "you must call it"
# mode travel with each request instead. mode="ANY" is what the old SDK's
# tool_config={"function_calling_config": "ANY"} meant: answer by calling
# submit_questions, never with prose.
GEMINI_CONFIG = types.GenerateContentConfig(
    tools=[types.Tool(function_declarations=[gemini_tool])],
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode="ANY")
    ),
)

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

# --- Asynchronous Client Provider ---
class AsyncClientProvider:
    """Provides lazily-initialized async clients."""
    def __init__(self):
        self._deepseek_client = None
        self._openai_client = None
        self._gemini_client = None

    @property
    def gemini(self):
        if self._gemini_client is None:
            api_key = current_app.config.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set in config.")
            self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

    @property
    def deepseek(self):
        if self._deepseek_client is None:
            api_key = current_app.config.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY not set in config.")
            self._deepseek_client = AsyncOpenAI(
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
            self._openai_client = AsyncOpenAI(api_key=api_key)
        return self._openai_client

# Instantiate the async provider
async_clients = AsyncClientProvider()