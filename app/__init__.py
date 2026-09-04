import os
from flask import Flask, jsonify
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