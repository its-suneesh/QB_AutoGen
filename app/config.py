# app/config.py

import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

def get_bool_env(var_name, default=False):
    """Helper to convert environment variable to boolean."""
    value = os.getenv(var_name, str(default)).lower()
    return value in ('true', '1', 't')

class Config:
    """Application configuration from environment variables."""

     # Flask Core Config
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY") 
    DEBUG = get_bool_env("FLASK_DEBUG", False)
    
    # JWT Configuration
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_ACCESS_TOKEN_EXPIRES_HOURS = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES_HOURS", "1"))
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=JWT_ACCESS_TOKEN_EXPIRES_HOURS)

    # Admin Credentials
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Model Name Configuration
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro-latest")
    DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo")

    RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI", "memcached://memcached:11211")

    # CORS Configuration
    CORS_ORIGINS = os.getenv("CORS_ORIGINS")

    # --- NEW: Configurable Logging Levels ---
    LOG_LEVEL_APP = os.getenv("LOG_LEVEL_APP", "INFO").upper()
    LOG_LEVEL_ERROR = os.getenv("LOG_LEVEL_ERROR", "ERROR").upper()
    LOG_LEVEL_ACCESS = os.getenv("LOG_LEVEL_ACCESS", "INFO").upper()
    LOG_LEVEL_SECURITY = os.getenv("LOG_LEVEL_SECURITY", "INFO").upper()


    # Critical variable check
    if not all([JWT_SECRET_KEY, ADMIN_USERNAME, ADMIN_PASSWORD, GOOGLE_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY]):
        raise ValueError("FATAL: Missing critical environment variables. Check .env file.")