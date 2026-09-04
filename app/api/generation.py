"""Writing new questions from a syllabus topic."""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request
from marshmallow import ValidationError

from app import usage
from app.schemas import GenerateSchema
from app.services import generate_questions_from_prompt_async

bp = Blueprint("generation", __name__)

access_logger = logging.getLogger("access")
app_logger = logging.getLogger("app")
# THIS SERVICE AUTHENTICATES NOBODY.
#
# Callers reach it through QuestionBankController.GenerateAiQuestions in the
# OnlineTCS .NET backend, which carries [Authorize] and so admits only a
# signed-in user holding the ordinary login token - the same gate as every other
# endpoint in the portal, the question-paper ones included.
#
# Two earlier arrangements were removed. A /login endpoint here authenticated
# one shared admin username and password, which the Angular app had to carry to
# use it - so the password shipped inside the JavaScript bundle every browser
# downloads, readable by anyone and enough to spend this service's LLM quota.
# After that, the backend's own token was checked here as well; that added no
# security over [Authorize], since it is the same token, but it required this
# service to keep a copy of the portal's JWT key, issuer and audience in step by
# hand - and any drift signed every teacher out of the portal the moment they
# pressed Generate, because the Angular interceptor logs out on a 401 from
# anywhere.
#
# ON EXPOSURE: with no check here, whatever can reach this service's address can
# spend its LLM quota. Keeping the portal the only caller is the deployment's
# job now - a firewall rule, a private network, or a reverse proxy that accepts
# only the backend - not this process's.


@bp.route('/generate_questions', methods=['POST'])
async def generate_questions_endpoint():

    validated_data = GenerateSchema().load(request.json)
    
    # Everything the request spends, across every rule it fans out to.
    with usage.collect() as spent:
        generated_questions = await generate_questions_from_prompt_async(validated_data)

    app_logger.info(
        f"Successfully generated {len(generated_questions)} questions."
    )
    return jsonify({
        "generated_questions": generated_questions,
        "usage": spent.as_dict(),
    })
