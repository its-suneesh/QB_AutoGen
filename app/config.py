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
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    # Only some Anthropic keys need this. An identity-linked key is not tied to
    # one workspace, so the API cannot tell which workspace to bill and refuses
    # the request with "anthropic-workspace-id is required when authenticating
    # with an identity-linked API key" until the id is sent. An ordinary
    # workspace key carries that in itself and needs nothing here, so this is
    # left empty unless the key demands it.
    ANTHROPIC_WORKSPACE_ID = os.getenv("ANTHROPIC_WORKSPACE_ID")

    # Model Name Configuration
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash")
    DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo")
    CLAUDE_MODEL_NAME = os.getenv("CLAUDE_MODEL_NAME", "claude-opus-5")

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

    # --- Textbook retrieval (RAG) ---
    #
    # Off by default: a deployment without a vector database keeps working
    # exactly as before, generating from book titles alone.
    RAG_ENABLED = get_bool_env("RAG_ENABLED", False)

    # Largest textbook /rag/index accepts. Flask refuses a bigger body
    # outright, so an oversized upload is never read into memory.
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024

    # Where the portal serves uploaded textbooks. Its /Photo path is mounted
    # ahead of the authentication middleware, so the PDFs are fetchable without
    # a token - which is what lets this work with no backend change.
    PORTAL_BASE_URL = os.getenv("PORTAL_BASE_URL", "")

    RAG_DATABASE_URL = os.getenv("RAG_DATABASE_URL", "")
    # Schema the index lives in. Its own, not public: since Postgres 15 an
    # ordinary login cannot create tables in public, but it can create a
    # schema of its own and own everything inside it. That turns a change
    # only a DBA could make into one the service makes for itself, and it
    # keeps a derived index out of a database shared with other things.
    RAG_DB_SCHEMA = os.getenv("RAG_DB_SCHEMA", "qbrag")
    # Seconds before an unreachable vector store is given up on. Retrieval
    # runs inside the generate request's own 180s ceiling, so failing open
    # is only useful if it also fails fast.
    RAG_DB_CONNECT_TIMEOUT = int(os.getenv("RAG_DB_CONNECT_TIMEOUT", "8"))
    RAG_DB_STATEMENT_TIMEOUT = int(os.getenv("RAG_DB_STATEMENT_TIMEOUT", "20"))
    # text-embedding-004 was retired by Google and now 404s on every API
    # version. gemini-embedding-001 is its stable replacement.
    RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "models/gemini-embedding-001")
    # Texts per embedding call. gemini-embedding-001 accepts far fewer per
    # request than text-embedding-004 did.
    RAG_EMBED_BATCH = int(os.getenv("RAG_EMBED_BATCH", "100"))
    RAG_FETCH_TIMEOUT = int(os.getenv("RAG_FETCH_TIMEOUT", "60"))
    # Self-signed certificates are normal on an internal portal host.
    RAG_VERIFY_TLS = get_bool_env("RAG_VERIFY_TLS", False)
    # --- Retrieval shape ---
    # Passages handed to the model. One passage is a contiguous stretch of a
    # book, not a paragraph, so this is smaller than it looks.
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
    # Pages taken per topic in the syllabus line before they are widened.
    RAG_PER_TOPIC = int(os.getenv("RAG_PER_TOPIC", "2"))
    # Pages kept either side of a hit, so a theorem keeps its proof and a
    # worked example keeps its solution instead of arriving cut in half.
    RAG_PAGE_WINDOW = int(os.getenv("RAG_PAGE_WINDOW", "1"))
    # Cosine distance past which a hit is treated as "this book does not cover
    # that topic". The prompt tells the model to base its questions on whatever
    # it is given, so off-topic pages are worse than no pages at all.
    #
    # 0.40 is measured, not guessed: with models/gemini-embedding-001 at 768
    # dimensions, pages of the right subject scored 0.25-0.28 and pages of a
    # different subject entirely scored 0.41-0.45. It sits in that gap, nearer
    # the top of it because the two errors are not equal - too loose lets a
    # wrong page through visibly, while too tight silently drops back to
    # generating from titles. When a topic is rejected the log records the
    # distance that missed, so this can be tuned from real books.
    RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.40"))
    # Ceiling on retrieved text per request, roughly 10k tokens.
    RAG_MAX_CHARS = int(os.getenv("RAG_MAX_CHARS", "55000"))

    # --- NEW: Configurable Logging Levels ---
    LOG_LEVEL_APP = os.getenv("LOG_LEVEL_APP", "INFO").upper()
    LOG_LEVEL_ERROR = os.getenv("LOG_LEVEL_ERROR", "ERROR").upper()
    LOG_LEVEL_ACCESS = os.getenv("LOG_LEVEL_ACCESS", "INFO").upper()
    LOG_LEVEL_SECURITY = os.getenv("LOG_LEVEL_SECURITY", "INFO").upper()


    # Critical variable check - the API keys, which are the only secrets left.
    #
    # ANTHROPIC_API_KEY is deliberately NOT in this list. Adding it would stop
    # every existing deployment from starting until someone put a Claude key in
    # its .env, for a provider those deployments may never ask for. A request
    # that does ask for "claude" without the key set fails on its own with a
    # readable message instead (see AsyncClientProvider.claude and
    # generate_questions_from_prompt_async); the other three stay mandatory
    # because a startup failure is the clearer signal for the providers that are
    # actually in use.
    if not all([GOOGLE_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY]):
        raise ValueError("FATAL: Missing critical environment variables. Check .env file.")