# app/logger.py

import os
import logging
from logging.handlers import RotatingFileHandler
import sys
import structlog

def setup_logging(app):
    """
    Configures a structured, JSON-based logging system.
    """
    log_directories = ["logs/app", "logs/error", "logs/access", "logs/security"]
    for directory in log_directories:
        os.makedirs(directory, exist_ok=True)

    # Common processors for all loggers
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]

    # Configure structlog to wrap the standard library logger
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # --- Create Handlers ---
    app_handler = RotatingFileHandler('logs/app/app.log', maxBytes=100000, backupCount=5)
    error_handler = RotatingFileHandler('logs/error/error.log', maxBytes=100000, backupCount=5)
    access_handler = RotatingFileHandler('logs/access/access.log', maxBytes=100000, backupCount=5)
    security_handler = RotatingFileHandler('logs/security/security.log', maxBytes=100000, backupCount=5)

    # --- Create Formatters ---
    # The renderer produces the final JSON output
    json_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    app_handler.setFormatter(json_formatter)
    error_handler.setFormatter(json_formatter)
    access_handler.setFormatter(json_formatter)
    security_handler.setFormatter(json_formatter)

    # --- Configure Root and Named Loggers ---
    # App logger (Flask's default)
    app.logger.addHandler(app_handler)
    app.logger.setLevel(logging.INFO)

    # Error logger
    error_logger = logging.getLogger('error')
    error_logger.addHandler(error_handler)
    error_logger.setLevel(logging.ERROR)

    # Access logger
    access_logger = logging.getLogger('access')
    access_logger.addHandler(access_handler)
    access_logger.setLevel(logging.INFO)

    # Security logger
    security_logger = logging.getLogger('security')
    security_logger.addHandler(security_handler)
    security_logger.setLevel(logging.INFO)


    app.logger.info("Structured logging configured.")

    # Return the raw loggers for use in the app
    return {
        "error": structlog.wrap_logger(error_logger),
        "access": structlog.wrap_logger(access_logger),
        "security": structlog.wrap_logger(security_logger)
    }