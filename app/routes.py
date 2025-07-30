# app/routes.py

import logging
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from marshmallow import ValidationError
import structlog

from .extensions import limiter
from .schemas import LoginSchema, GenerateSchema
from .services import generate_questions_from_prompt

main_bp = Blueprint('main', __name__)

# Use structlog for structured logging
access_logger = structlog.get_logger('access')
security_logger = structlog.get_logger('security')
app_logger = structlog.get_logger(__name__)


@main_bp.before_app_request
def log_request_info():
    """Logs every incoming request to the access log."""
    if request.path != '/':
        access_logger.info(
            "incoming_request",
            method=request.method,
            path=request.path,
            remote_addr=request.remote_addr,
            user_agent=request.user_agent.string
        )

@main_bp.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
def handle_all(path):
    app_logger.warning(
        "unmatched_route_accessed",
        method=request.method,
        path=f"/{path}",
        remote_addr=request.remote_addr
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

@main_bp.route("/", methods=["GET"])
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
    try:
        data = LoginSchema().load(request.json)
        username = data["username"]
        password = data["password"]

        # Use a secure way to check credentials
        if username == current_app.config["ADMIN_USERNAME"] and password == current_app.config["ADMIN_PASSWORD"]:
            access_token = create_access_token(identity=username)
            security_logger.info(
                "login_success",
                username=username,
                remote_addr=request.remote_addr
            )
            return jsonify(access_token=access_token)
        else:
            # Avoid logging the username if it's invalid to prevent user enumeration
            security_logger.warning(
                "login_failed",
                remote_addr=request.remote_addr,
                reason="Invalid credentials"
            )
            return jsonify({"error": "Bad username or password"}), 401

    except ValidationError as err:
        raise err

@main_bp.route('/generate_questions', methods=['POST'])
@jwt_required()
def generate_questions_endpoint():
    """
    The main API endpoint to handle question generation requests.
    This endpoint is protected and requires a valid JWT.
    """
    user_identity = get_jwt_identity()
    app_logger.info(
        "generate_endpoint_accessed",
        user=user_identity
    )
    
    validated_data = GenerateSchema().load(request.json)
    
    generated_questions = generate_questions_from_prompt(validated_data)

    app_logger.info(
        "questions_generated_successfully",
        question_count=len(generated_questions)
    )
    return jsonify({"generated_questions": generated_questions})