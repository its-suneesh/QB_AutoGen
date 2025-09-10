import logging
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from marshmallow import ValidationError

from .extensions import limiter
from .schemas import LoginSchema, GenerateSchema
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
            "/login",
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

@main_bp.route("/login", methods=["POST"])
@limiter.limit("5 per minute")
def login():
    """Authenticates a user and returns a JWT."""
    # The global error handler in __init__.py will catch any ValidationError
    data = LoginSchema().load(request.json)
    username = data["username"]
    password = data["password"]

    if username == current_app.config["ADMIN_USERNAME"] and password == current_app.config["ADMIN_PASSWORD"]:
        access_token = create_access_token(identity=username)
        security_logger.info(
            f"Login success for user '{username}' from {request.remote_addr}"
        )
        return jsonify(access_token=access_token)
    else:
        security_logger.warning(
            f"Login failed: Invalid credentials from {request.remote_addr}"
        )
        return jsonify({"error": "Bad username or password"}), 401


@main_bp.route('/generate_questions', methods=['POST'])
@jwt_required()
async def generate_questions_endpoint():
    user_identity = get_jwt_identity()
    app_logger.info(f"Generation endpoint accessed by user '{user_identity}'")
    
    validated_data = GenerateSchema().load(request.json)
    
    generated_questions = await generate_questions_from_prompt_async(validated_data)

    app_logger.info(
        f"Successfully generated {len(generated_questions)} questions."
    )
    return jsonify({"generated_questions": generated_questions})