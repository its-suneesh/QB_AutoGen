# app/logger.py

import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(app):
    """Configures a structured logging system with separate files."""
    log_directories = ["logs/app", "logs/error", "logs/access", "logs/security"]
    for directory in log_directories:
        os.makedirs(directory, exist_ok=True)

    # General Application Logger (attached to app.logger)
    app_handler = RotatingFileHandler('logs/app/app.log', maxBytes=100000, backupCount=5)
    app_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    app.logger.addHandler(app_handler)
    app.logger.setLevel(logging.INFO)

    # Error Logger
    error_handler = RotatingFileHandler('logs/error/error.log', maxBytes=100000, backupCount=5)
    error_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    error_logger = logging.getLogger('error')
    error_logger.addHandler(error_handler)
    error_logger.setLevel(logging.ERROR)

    # Access Logger
    access_handler = RotatingFileHandler('logs/access/access.log', maxBytes=100000, backupCount=5)
    access_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    access_logger = logging.getLogger('access')
    access_logger.addHandler(access_handler)
    access_logger.setLevel(logging.INFO)

    # Security Logger
    security_handler = RotatingFileHandler('logs/security/security.log', maxBytes=100000, backupCount=5)
    security_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    security_logger = logging.getLogger('security')
    security_logger.addHandler(security_handler)
    security_logger.setLevel(logging.INFO)

    app.logger.info("Logging successfully configured.")
    return {
        "error": error_logger,
        "access": access_logger,
        "security": security_logger
    }
