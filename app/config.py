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
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY") # Used for session signing
    DEBUG = get_bool_env("FLASK_DEBUG", False)
    
    # JWT Configuration
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_ACCESS_TOKEN_EXPIRES_HOURS = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES_HOURS", "1"))
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=JWT_ACCESS_TOKEN_EXPIRES_HOURS)

    # Admin Credentials
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

    # --- MODIFIED: API Keys ---
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # <-- ADDED

    # --- ADDED: Model Name Configuration ---
    # Set defaults or leave as None to be caught by the check below
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro-latest")
    DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo")

    RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI", "memory://")

    # CORS Configuration
    CORS_ORIGINS = os.getenv("CORS_ORIGINS")

    # --- MODIFIED: Critical variable check ---
    if not all([JWT_SECRET_KEY, ADMIN_USERNAME, ADMIN_PASSWORD, GOOGLE_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY]): # <-- ADDED OPENAI_API_KEY
        raise ValueError("FATAL: Missing critical environment variables. Check .env file.")