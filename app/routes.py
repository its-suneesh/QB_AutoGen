import logging
from flask import Blueprint, request, jsonify, current_app
from marshmallow import ValidationError
from .schemas import GenerateSchema
from .services import generate_questions_from_prompt_async

main_bp = Blueprint('main', __name__)

access_logger = logging.getLogger('access')
security_logger = logging.getLogger('security')
app_logger = logging.getLogger('app')


@main_bp.before_app_request
def log_request_info():
    """Logs every incoming request to the access log."""
    if request.path != '/':
        access_logger.info(
            f"Incoming request: {request.method} {request.path} "
            f"from {request.remote_addr} | User-Agent: {request.user_agent.string}"
        )

@main_bp.route("/", methods=["GET"])
def index():
    """Root endpoint listing the available API endpoints."""
    return jsonify({
        "service": "QB AutoGen API",
        "message": "Dynamic Question Generation API",
        "available_endpoints": [
            "/",
            "/health",
                "/generate_questions"
        ]
    }), 200

@main_bp.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
def handle_all(path):
    app_logger.info(
        f"Unmatched route accessed: {request.method} /{path} "
        f"from {request.remote_addr}"
    )
    return jsonify({
        "error": "Not Found",
        "message": f"The requested endpoint '/{path}' does not exist",
        "available_endpoints": [
            "/",
                "/generate_questions"
        ]
    }), 404

@main_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for Docker and load balancers."""
    return jsonify({
        "status": "healthy",
        "service": "QB AutoGen API",
        "version": "1.0.0"
    }), 200

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


@main_bp.route('/generate_questions', methods=['POST'])
async def generate_questions_endpoint():

    validated_data = GenerateSchema().load(request.json)
    
    generated_questions = await generate_questions_from_prompt_async(validated_data)

    app_logger.info(
        f"Successfully generated {len(generated_questions)} questions."
    )
    return jsonify({"generated_questions": generated_questions})