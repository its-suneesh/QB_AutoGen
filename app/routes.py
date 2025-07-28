# app/routes.py

import logging
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from marshmallow import ValidationError

from .extensions import limiter
from .schemas import LoginSchema, GenerateSchema
from .services import generate_questions_from_prompt

main_bp = Blueprint('main', __name__)

access_logger = logging.getLogger('access')
security_logger = logging.getLogger('security')

@main_bp.before_app_request
def log_request_info():
    """Logs every incoming request to the access log."""
    access_logger.info(f'Request: {request.method} {request.path} from {request.remote_addr}')

@main_bp.route("/login", methods=["POST"])
@limiter.limit("5 per minute")
def login():
    """Authenticates a user and returns a JWT."""
    try:
        data = LoginSchema().load(request.json)
        username = data["username"]
        password = data["password"]

        if username == current_app.config["ADMIN_USERNAME"] and password == current_app.config["ADMIN_PASSWORD"]:
            access_token = create_access_token(identity=username)
            security_logger.info(f"Successful login for user: '{username}' from {request.remote_addr}")
            return jsonify(access_token=access_token)
        else:
            security_logger.warning(f"Failed login attempt for user: '{username}' from {request.remote_addr}")
            return jsonify({"error": "Bad username or password"}), 401

    except ValidationError as err:
        # This will be caught by the global error handler in __init__.py
        raise err


@main_bp.route('/generate_questions', methods=['POST'])
@jwt_required()
def generate_questions_endpoint():
    """
    The main API endpoint to handle question generation requests.
    This endpoint is protected and requires a valid JWT.
    """
    current_app.logger.info(f"'/generate' endpoint accessed by user: '{get_jwt_identity()}'")
    
    # validated_data will now contain the 'model' field from the schema
    validated_data = GenerateSchema().load(request.json)
    
    generated_questions = generate_questions_from_prompt(validated_data)

    current_app.logger.info(f"Successfully generated {len(generated_questions)} questions.")
    return jsonify({"generated_questions": generated_questions})