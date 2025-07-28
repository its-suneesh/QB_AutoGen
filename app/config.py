# app/config.py

import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration from environment variables."""
    
    # JWT Configuration
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)

    # Admin Credentials
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

    # Google API Key
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # DeepSeek API Key
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

    # CORS Configuration
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

    # Critical variable check
    if not all([JWT_SECRET_KEY, ADMIN_USERNAME, ADMIN_PASSWORD, GOOGLE_API_KEY, DEEPSEEK_API_KEY]):
        raise ValueError("FATAL: Missing critical environment variables. Check .env file.")