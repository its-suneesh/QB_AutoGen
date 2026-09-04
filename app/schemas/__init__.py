"""Request and tool-output shapes, and the parsing that makes them survive.

A model asked for JSON does not always send JSON - see _loads_latex_json, which
reads an answer whose LaTeX has broken its own quoting rather than discarding a
full set of questions over punctuation.
"""

from .generation import (
    BookDetailsSchema,
    GenerateSchema,
    LLMToolOutputSchema,
    RuleSchema,
    _loads_latex_json,
)

__all__ = [
    "BookDetailsSchema",
    "GenerateSchema",
    "LLMToolOutputSchema",
    "RuleSchema",
    "_loads_latex_json",
]
