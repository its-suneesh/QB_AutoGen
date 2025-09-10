import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask

def setup_logging(app: Flask):
    """
    Configures a simple, standard logging system that outputs to both
    the console (stdout) and rotating files.
    """
    log_directories = ["logs/app", "logs/error", "logs/access", "logs/security"]
    for directory in log_directories:
        os.makedirs(directory, exist_ok=True)

    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    log_configs = {
        'access': {
            'level': app.config.get("LOG_LEVEL_ACCESS", "INFO"),
            'filename': 'logs/access/access.log'
        },
        'error': {
            'level': app.config.get("LOG_LEVEL_ERROR", "ERROR"),
            'filename': 'logs/error/error.log'
        },
        'security': {
            'level': app.config.get("LOG_LEVEL_SECURITY", "INFO"),
            'filename': 'logs/security/security.log'
        }
    }

    # --- Configure the root/app logger (Flask's default) ---
    app.logger.handlers.clear() # Remove any default handlers
    app.logger.setLevel(app.config.get("LOG_LEVEL_APP", "INFO"))

    # Add console handler for the app logger
    app_console_handler = logging.StreamHandler(sys.stdout)
    app_console_handler.setFormatter(log_formatter)
    app.logger.addHandler(app_console_handler)

    # Add file handler for the app logger
    app_file_handler = RotatingFileHandler(
        'logs/app/app.log', maxBytes=100000, backupCount=5, encoding='utf-8'
    )
    app_file_handler.setFormatter(log_formatter)
    app.logger.addHandler(app_file_handler)

    # --- Configure the other named loggers ---
    for name, config in log_configs.items():
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.setLevel(config['level'])
        logger.propagate = False 

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)

        # Add file handler
        file_handler = RotatingFileHandler(
            config['filename'], maxBytes=100000, backupCount=5, encoding='utf-8'
        )
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    app.logger.info("Simple logging configured to output to stdout and files.")