"""What the service actually does, once HTTP has been dealt with.

Nothing here reads a Flask request or writes a response - these take plain
arguments and return plain data, so they can be tested without a web server and
called from somewhere else later.

  generation  writes new questions from a syllabus topic
  extraction  files questions the teacher already has
"""

from .errors import ServiceError
from .extraction import (
    SUPPORTED_MODELS,
    classify_questions,
)
from .generation import generate_questions_from_prompt_async

__all__ = [
    "ServiceError",
    "SUPPORTED_MODELS",
    "classify_questions",
    "generate_questions_from_prompt_async",
]
