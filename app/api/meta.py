"""What this service is, whether it is up, and where it is not."""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from app.config import Config
from app.services import ServiceError

bp = Blueprint("meta", __name__)

access_logger = logging.getLogger("access")
app_logger = logging.getLogger("app")
error_logger = logging.getLogger("error")
@bp.before_app_request
def log_request_info():
    """Logs every incoming request to the access log."""
    if request.path != '/':
        access_logger.info(
            f"Incoming request: {request.method} {request.path} "
            f"from {request.remote_addr} | User-Agent: {request.user_agent.string}"
        )



@bp.route("/", methods=["GET"])
def index():
    """Root endpoint listing the available API endpoints."""
    return jsonify({
        "service": "QB AutoGen API",
        "message": "Dynamic Question Generation API",
        "available_endpoints": [
            "/",
            "/health",
            "/ready",
            "/generate_questions",
            "/classify_questions",
            "/rag/index",
            "/rag/prune",
            "/rag/status/<paper_id>"
        ]
    }), 200



@bp.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
def handle_all(path):
    app_logger.info(
        f"Unmatched route accessed: {request.method} /{path} "
        f"from {request.remote_addr}"
    )
    return jsonify({
        "error": "Not Found",
        "message": f"The requested endpoint '/{path}' does not exist",
        "available_endpoints": [
            "/",
            "/generate_questions",
            "/classify_questions",
            "/rag/index",
            "/rag/prune",
            "/rag/status/<paper_id>"
        ]
    }), 404



@bp.route("/health", methods=["GET"])
def health_check():
    """
    Is the process alive. Deliberately checks nothing else.

    This is what Docker restarts on, so it must not fail for a reason that
    restarting cannot fix - a database outage would otherwise put the container
    in a restart loop while the service itself was fine.
    """
    return jsonify({
        "status": "healthy",
        "service": "QB AutoGen API",
        "version": "1.0.0"
    }), 200


@bp.route("/ready", methods=["GET"])
def readiness_check():
    """
    Can the service actually do its work.

    /health answers 200 whatever is wrong with the world, which is right for a
    restart signal and useless for anything else: while it was the only check,
    a retired API key, an unreachable database and a missing pgvector extension
    - all three of which have happened - reported the service as healthy.

    So this one is allowed to fail. It reports 503 when something is broken,
    names what, and says whether question generation still works, because that
    keeps running on titles alone when retrieval is down.
    """
    checks = {}

    # A model key, or the service cannot answer any request at all.
    configured = [name for name, key in (
        ("claude", Config.ANTHROPIC_API_KEY),
        ("gemini", Config.GOOGLE_API_KEY),
    ) if key]
    checks["models"] = {
        "ok": bool(configured),
        "detail": ", ".join(configured) if configured else "no model API key is set",
    }

    # The textbook index. Its absence degrades generation rather than stopping
    # it, so it is reported apart from the rest.
    if not Config.RAG_ENABLED:
        checks["retrieval"] = {"ok": True, "detail": "disabled"}
    else:
        try:
            from app import retrieval
            retrieval.document_status(0)
            checks["retrieval"] = {"ok": True, "detail": "reachable"}
        except Exception as exc:
            checks["retrieval"] = {
                "ok": False,
                "detail": str(exc).strip().splitlines()[0][:120],
            }

    # Retrieval failing open means questions can still be generated from titles.
    ready = checks["models"]["ok"]

    return jsonify({
        "ready": ready,
        "generation": "available" if ready else "unavailable",
        "checks": checks,
    }), (200 if ready else 503)
