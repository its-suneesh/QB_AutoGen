"""Filing questions the teacher already has.

The reverse of generation: nothing is invented, and what comes back is a
proposal for the teacher to correct rather than an answer.
"""

from __future__ import annotations

import json
import logging

from flask import Blueprint, jsonify, request

from app import usage
from app.services import SUPPORTED_MODELS, classify_questions


MAX_QUESTIONS = 300

bp = Blueprint("extraction", __name__)

app_logger = logging.getLogger("app")
@bp.route('/classify_questions', methods=['POST'])
async def classify_questions_endpoint():
    """
    Files questions the teacher already has against their course.

    JSON:
        questions   - the question texts, in the order they were written
        vocabulary  - the lists the classification must choose from:
                      modules, units, question_types, difficulty_levels,
                      cognitive_levels, course_outcomes
        model       - which provider does the work (default "claude")

    The text is returned unchanged; only the labels are the model's.
    """
    body = request.get_json(silent=True) or {}
    questions = body.get('questions')

    if not isinstance(questions, list):
        return jsonify({"error": "Bad Request",
                        "message": "questions must be a list."}), 400
    if not questions:
        return jsonify({"error": "Bad Request",
                        "message": "No questions were sent."}), 400
    if len(questions) > MAX_QUESTIONS:
        return jsonify({
            "error": "Payload Too Large",
            "message": f"At most {MAX_QUESTIONS} questions per upload. "
                       f"This file has {len(questions)}. Split it and try again.",
        }), 413

    vocabulary = body.get('vocabulary') or {}
    if not isinstance(vocabulary, dict):
        return jsonify({"error": "Bad Request",
                        "message": "vocabulary must be a JSON object."}), 400

    model = (body.get('model') or 'claude').strip().lower()
    if model not in SUPPORTED_MODELS:
        return jsonify({
            "error": "Bad Request",
            "message": f"'{model}' cannot classify questions yet. "
                       "Supported: " + ", ".join(SUPPORTED_MODELS),
        }), 400

    try:
        with usage.collect() as spent:
            result = await classify_questions(questions, vocabulary, model)
        result["usage"] = spent.as_dict()
    except Exception as exc:
        app_logger.exception("Question classification failed")
        return jsonify({"error": "Classification Failed", "message": str(exc)}), 500

    app_logger.info("Classified %s question(s)", len(result["questions"]))
    return jsonify(result), 200
