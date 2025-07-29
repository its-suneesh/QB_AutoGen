# app/__init__.py

import os
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from marshmallow import ValidationError

from .config import Config
from .logger import setup_logging
from .extensions import jwt, limiter, genai
from .routes import main_bp
# --- ADDED ---
# Import the custom exception to handle it globally
from .services import ServiceError

def create_app():
    """Application factory function."""
    load_dotenv()
    
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize CORS
    CORS(app, resources={r"/*": {"origins": app.config["CORS_ORIGINS"]}})

    # Initialize logging
    loggers = setup_logging(app)
    app.loggers = loggers # Attach loggers to app context if needed elsewhere

    # Initialize extensions
    jwt.init_app(app)
    limiter.init_app(app)

    # Configure Gemini API
    try:
        genai.configure(api_key=app.config['GOOGLE_API_KEY'])
        app.logger.info("Gemini API configured successfully.")
    except Exception as e:
        app.logger.critical(f"Error configuring Gemini API: {e}")
        exit(f"Could not configure Gemini API: {e}")

    # Register blueprints
    app.register_blueprint(main_bp)

    # Register global error handlers
    @app.errorhandler(ValidationError)
    def handle_marshmallow_validation(err):
        app.logger.warning(f"Validation Error: {err.messages} for request")
        return jsonify({"error": "Validation Error", "messages": err.messages}), 400

    # --- ADDED ---
    # New error handler for service-level failures (e.g., API calls)
    @app.errorhandler(ServiceError)
    def handle_service_error(e):
        # The error is already logged in the service layer where it occurred
        return jsonify({"error": "Service Error", "message": str(e)}), e.status_code

    @app.errorhandler(Exception)
    def handle_generic_exception(e):
        loggers["error"].error(f"Unhandled Exception: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

    return app