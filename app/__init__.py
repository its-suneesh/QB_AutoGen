import os
import secrets
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from marshmallow import ValidationError
from werkzeug.exceptions import HTTPException
import logging # --- ADDED ---

from .config import Config
from .logger import setup_logging
from .extensions import async_clients
from .api import register as register_routes
from .services import ServiceError

def create_app():
    load_dotenv()
    
    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(app, resources={
        r"/*": {
            "origins": app.config["CORS_ORIGINS"],
            "allow_headers": "*", 
            "expose_headers": "*", 
        }
    })
    
    # This will now set up the simple logging system
    setup_logging(app)

    # Built here rather than on first use so a bad key fails at startup, the way
    # the old genai.configure() call did. The provider reads it from
    # current_app.config, hence the context.
    try:
        with app.app_context():
            async_clients.gemini
        app.logger.info("Gemini API configured successfully.")
    except Exception as e:
        app.logger.critical(f"Error configuring Gemini API: {e}", exc_info=True)
        exit(f"Could not configure Gemini API: {e}")

    # Who may use this service.
    #
    # One check for every route, rather than a decorator per endpoint that a
    # new endpoint can forget to carry. The probes stay open: /health is what
    # Docker restarts on and /ready is what a monitor reads, and neither should
    # need a secret to answer.
    OPEN_PATHS = {"/health", "/ready"}

    @app.before_request
    def require_api_key():
        if not app.config.get("API_KEY"):
            return None
        if request.method == "OPTIONS" or request.path in OPEN_PATHS:
            return None

        presented = request.headers.get("X-API-Key", "")
        # compare_digest, not ==: a plain comparison returns as soon as two
        # characters differ, and the time it takes tells a guesser how much of
        # the key they have right.
        if not presented or not secrets.compare_digest(presented, app.config["API_KEY"]):
            logging.getLogger("access").warning(
                "Rejected %s %s - no valid X-API-Key", request.method, request.path)
            return jsonify({"error": "Unauthorized", "message": "Invalid or missing API key"}), 401

        return None

    register_routes(app)

    @app.errorhandler(ValidationError)
    def handle_marshmallow_validation(err):
        app.logger.warning(f"Validation Error: {err.messages}")
        return jsonify({"error": "Validation Error", "messages": err.messages}), 400

    @app.errorhandler(ServiceError)
    def handle_service_error(e):
        return jsonify({"error": "Service Error", "message": str(e)}), e.status_code

    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        return jsonify({"error": e.name, "message": e.description}), e.code

    @app.errorhandler(Exception)
    def handle_generic_exception(e):
        # --- MODIFIED: Get logger by name and log the exception ---
        logging.getLogger('error').error(f"Unhandled Exception: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

    return app