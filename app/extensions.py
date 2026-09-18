from anthropic import AsyncAnthropic
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
                        "answer_latex": {"type": "STRING"},
                        "unit_id": {"type": "STRING"},
                        "co_id": {"type": "STRING"},
                        "cognitive_level_id": {"type": "STRING"},
                        "difficulty_id": {"type": "STRING"}
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
                            "answer_latex": {"type": "string"},
                        "unit_id": {"type": "string"},
                        "co_id": {"type": "string"},
                        "cognitive_level_id": {"type": "string"},
                        "difficulty_id": {"type": "string"}
                        },
                        "required": ["question", "answer", "question_latex", "answer_latex"]
                    }
                }
            },
            "required": ["questions"]
        }
    }
}

# The same contract a third time, in Anthropic's shape.
#
# Claude takes a JSON Schema under "input_schema" - not "parameters" as OpenAI
# does, and not Gemini's upper-case type names - so the three definitions cannot
# be one shared dict however alike they read.
CLAUDE_TOOL = {
    "name": "submit_questions",
    "description": (
        "Submits a list of generated questions. Call this exactly once, with "
        "every question that was asked for."
    ),
    "input_schema": {
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
                        "answer_latex": {"type": "string"},
                        "unit_id": {"type": "string"},
                        "co_id": {"type": "string"},
                        "cognitive_level_id": {"type": "string"},
                        "difficulty_id": {"type": "string"}
                    },
                    "required": ["question", "answer", "question_latex", "answer_latex"]
                }
            }
        },
        "required": ["questions"]
    }
}

# Answer by calling submit_questions, never with prose - what mode="ANY" does
# for Gemini above. disable_parallel_tool_use holds it to the single call
# _call_provider reads; left off, a large batch can arrive split across several
# tool_use blocks in the one response and everything after the first is dropped.
CLAUDE_TOOL_CHOICE = {
    "type": "tool",
    "name": "submit_questions",
    "disable_parallel_tool_use": True,
}

# --- Asynchronous Client Provider ---
class AsyncClientProvider:
    """Provides lazily-initialized async clients."""
    def __init__(self):
        self._deepseek_client = None
        self._openai_client = None
        self._gemini_client = None
        self._claude_client = None

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

    @property
    def claude(self):
        # The key is passed explicitly rather than left to AsyncAnthropic's own
        # environment lookup, so the client is built from the same .env-backed
        # config as the other three and a missing key is reported here by name
        # instead of failing later inside the SDK.
        if self._claude_client is None:
            api_key = current_app.config.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in config.")
            # The SDK sets this header itself only for its Bedrock and
            # credential-provider clients, never for a plain api_key one, so an
            # identity-linked key has to be given the workspace here or every
            # request comes back 400. Sent only when configured: an ordinary
            # workspace key does not need it.
            workspace_id = current_app.config.get("ANTHROPIC_WORKSPACE_ID")
            headers = (
                {"anthropic-workspace-id": workspace_id} if workspace_id else None
            )
            self._claude_client = AsyncAnthropic(
                api_key=api_key,
                default_headers=headers,
            )
        return self._claude_client

# Instantiate the async provider
async_clients = AsyncClientProvider()