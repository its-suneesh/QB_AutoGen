# app/config.py

import os
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
    
    # No authentication settings.
    #
    # This service authenticates nobody. Callers reach it through
    # QuestionBankController.GenerateAiQuestions in the OnlineTCS .NET backend,
    # which carries [Authorize] and admits only a signed-in user holding the
    # ordinary login token - the same gate as every other endpoint in the
    # portal. There was a /login here once, guarded by one shared admin
    # password; the Angular app had to carry that password to use it, so it
    # shipped inside the JavaScript bundle where anyone could read it. Then
    # there was a check against the backend's own token, which added no security
    # over [Authorize] but required this service to keep a copy of the portal's
    # JWT key, issuer and audience in step by hand - and any drift signed every
    # teacher out of the portal the moment they pressed Generate.
    #
    # Consequence: whatever can reach this service's address can spend its LLM
    # quota. Keeping the portal the only caller is now the deployment's job - a
    # firewall rule, a private network, or a reverse proxy that accepts only the
    # backend.

    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Model Name Configuration
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash")
    DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo")

    # Which sites a BROWSER may call this service from. "*" for any.
    #
    # A list is split on commas here; passing the raw string through meant a
    # value like "http://a,http://b" was offered to flask-cors as one origin
    # named "http://a,http://b", which matched neither of them.
    #
    # CORS is not access control. It tells a browser whether a page from
    # another origin may READ the response; it stops nothing that is not a
    # browser, so curl, a script or another server reaches this service whatever
    # is set here. It also plays no part in the portal's own use of it: that
    # goes browser -> .NET backend -> here, and the last hop is server to
    # server, where no Origin header exists and CORS never applies.
    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ] or "*"

    # Where this service listens.
    #
    # These are the only place the address lives. run.py binds them, the
    # Dockerfile's healthcheck and docker-compose read the same PORT out of
    # .env, so moving the service to another port at deploy time is one edit in
    # .env and nothing else. HOST stays on the loopback by default because the
    # service has no authentication of its own (see the note above) - inside a
    # container docker-compose sets HOST=0.0.0.0, which it must to be reachable.
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "9000"))
    WORKERS = int(os.getenv("WORKERS", "1"))

    # --- NEW: Configurable Logging Levels ---
    LOG_LEVEL_APP = os.getenv("LOG_LEVEL_APP", "INFO").upper()
    LOG_LEVEL_ERROR = os.getenv("LOG_LEVEL_ERROR", "ERROR").upper()
    LOG_LEVEL_ACCESS = os.getenv("LOG_LEVEL_ACCESS", "INFO").upper()
    LOG_LEVEL_SECURITY = os.getenv("LOG_LEVEL_SECURITY", "INFO").upper()


    # Critical variable check - the API keys, which are the only secrets left.
    if not all([GOOGLE_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY]):
        raise ValueError("FATAL: Missing critical environment variables. Check .env file.")