import json
import logging
from flask import Blueprint, request, jsonify, current_app
from marshmallow import ValidationError
from .schemas import GenerateSchema
from .services import generate_questions_from_prompt_async

main_bp = Blueprint('main', __name__)

access_logger = logging.getLogger('access')
security_logger = logging.getLogger('security')
app_logger = logging.getLogger('app')

# Most a single upload may carry. A spreadsheet of more than this is a year's
# question bank rather than one paper, and every question costs a model call's
# share of the batch it sits in.
MAX_QUESTIONS = 300


@main_bp.before_app_request
def log_request_info():
    """Logs every incoming request to the access log."""
    if request.path != '/':
        access_logger.info(
            f"Incoming request: {request.method} {request.path} "
            f"from {request.remote_addr} | User-Agent: {request.user_agent.string}"
        )

@main_bp.route("/", methods=["GET"])
def index():
    """Root endpoint listing the available API endpoints."""
    return jsonify({
        "service": "QB AutoGen API",
        "message": "Dynamic Question Generation API",
        "available_endpoints": [
            "/",
            "/health",
            "/generate_questions",
            "/extract_questions",
            "/classify_questions",
            "/rag/index",
            "/rag/prune",
            "/rag/status/<paper_id>"
        ]
    }), 200

@main_bp.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
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
            "/extract_questions",
            "/classify_questions",
            "/rag/index",
            "/rag/prune",
            "/rag/status/<paper_id>"
        ]
    }), 404

@main_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for Docker and load balancers."""
    return jsonify({
        "status": "healthy",
        "service": "QB AutoGen API",
        "version": "1.0.0"
    }), 200

# THIS SERVICE AUTHENTICATES NOBODY.
#
# Callers reach it through QuestionBankController.GenerateAiQuestions in the
# OnlineTCS .NET backend, which carries [Authorize] and so admits only a
# signed-in user holding the ordinary login token - the same gate as every other
# endpoint in the portal, the question-paper ones included.
#
# Two earlier arrangements were removed. A /login endpoint here authenticated
# one shared admin username and password, which the Angular app had to carry to
# use it - so the password shipped inside the JavaScript bundle every browser
# downloads, readable by anyone and enough to spend this service's LLM quota.
# After that, the backend's own token was checked here as well; that added no
# security over [Authorize], since it is the same token, but it required this
# service to keep a copy of the portal's JWT key, issuer and audience in step by
# hand - and any drift signed every teacher out of the portal the moment they
# pressed Generate, because the Angular interceptor logs out on a 401 from
# anywhere.
#
# ON EXPOSURE: with no check here, whatever can reach this service's address can
# spend its LLM quota. Keeping the portal the only caller is the deployment's
# job now - a firewall rule, a private network, or a reverse proxy that accepts
# only the backend - not this process's.


@main_bp.route('/generate_questions', methods=['POST'])
async def generate_questions_endpoint():

    validated_data = GenerateSchema().load(request.json)
    
    generated_questions = await generate_questions_from_prompt_async(validated_data)

    app_logger.info(
        f"Successfully generated {len(generated_questions)} questions."
    )
    return jsonify({"generated_questions": generated_questions})

# --- Textbook indexing -----------------------------------------------------
#
# The portal posts the PDF here when a textbook is uploaded, and gets back
# whether it can actually be used for question generation.
#
# The file is sent as bytes rather than as a link on purpose: this service then
# never needs the portal's address, never reaches back into it, and does not
# depend on those uploads staying publicly readable.


@main_bp.route('/rag/index', methods=['POST'])
def rag_index_endpoint():
    """
    multipart/form-data:
        file      - the PDF                        (required)
        doc_id    - e.g. "1_734"                   (required)
        BookName  - title, for citations           (optional)
        BookType  - "Textbook" / "Reference"       (optional)
    """
    from app import rag
    from app.config import Config

    if not Config.RAG_ENABLED:
        return jsonify({
            "status": "disabled",
            "message": "Textbook indexing is not enabled on this server.",
        }), 503

    upload = request.files.get('file')
    if upload is None or not upload.filename:
        return jsonify({"error": "Bad Request", "message": "No file was sent."}), 400

    doc_id = (request.form.get('doc_id') or '').strip()
    parsed = rag.parse_doc_id(doc_id or upload.filename)
    if not parsed:
        return jsonify({
            "error": "Bad Request",
            "message": "doc_id must look like '1_734' (slot_paper).",
        }), 400
    doc_id, paper_id, slot_no = parsed

    pdf_bytes = upload.read()
    if not pdf_bytes:
        return jsonify({"error": "Bad Request", "message": "The file was empty."}), 400
    try:
        result = rag.index_pdf_bytes(
            pdf_bytes, doc_id, paper_id, slot_no,
            request.form.get('BookName', ''), request.form.get('BookType', ''),
        )
    except Exception as exc:
        app_logger.exception("RAG index failed for %s", doc_id)
        return jsonify({
            "error": "Indexing Failed",
            "message": str(exc),
            "doc_id": doc_id,
        }), 500

    app_logger.info("RAG indexed %s -> %s", doc_id, result.get("status"))
    return jsonify(result), 200


@main_bp.route('/rag/prune', methods=['POST'])
def rag_prune_endpoint():
    """
    Makes the index match a course's current book list.

    JSON: {"paper_id": 734,
           "books": [{"doc_id": "1_734", "BookName": ..., "BookType": ...}, ...]}

    The portal sends its whole list on every save; anything indexed for that
    course and not named here is dropped. An empty list is legitimate - it means
    the course has no books left.
    """
    from app import rag
    from app.config import Config

    if not Config.RAG_ENABLED:
        return jsonify({"status": "disabled", "removed": []}), 503

    body = request.get_json(silent=True) or {}
    try:
        paper_id = int(body.get("paper_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "Bad Request", "message": "paper_id is required."}), 400

    books = body.get("books")
    if not isinstance(books, list):
        # Absent is NOT the same as empty: an empty list deletes the course's
        # whole index, so it has to be sent deliberately rather than by omission.
        return jsonify({
            "error": "Bad Request",
            "message": "books must be a list (send [] to clear the course).",
        }), 400

    try:
        result = rag.prune_documents(paper_id, books)
    except Exception as exc:
        app_logger.exception("RAG prune failed for paper %s", paper_id)
        return jsonify({"error": "Prune Failed", "message": str(exc)}), 500

    return jsonify(result), 200


@main_bp.route('/rag/status/<int:paper_id>', methods=['GET'])
def rag_status_endpoint(paper_id: int):
    """Index status of every book of one course, so the portal can show it."""
    from app import rag
    from app.config import Config

    if not Config.RAG_ENABLED:
        return jsonify({"status": "disabled", "books": []}), 503

    try:
        return jsonify({"paper_id": paper_id, "books": rag.document_status(paper_id)}), 200
    except Exception as exc:
        app_logger.exception("RAG status failed for paper %s", paper_id)
        return jsonify({"error": "Status Unavailable", "message": str(exc)}), 500


@main_bp.route('/extract_questions', methods=['POST'])
async def extract_questions_endpoint():
    """
    Reads the questions out of an uploaded question paper.

    multipart/form-data:
        file        - the PDF                                     (required)
        vocabulary  - JSON: the lists the classification must choose from,
                      {"modules": [...], "units": [...],
                       "question_types": [...], "difficulty_levels": [...],
                       "cognitive_levels": [...], "course_outcomes": [...]}
        model       - which model reads the paper (default "claude")

    Needs no database and writes nothing: the paper is read once and discarded.
    """
    from app import extract, rag

    upload = request.files.get('file')
    if upload is None or not upload.filename:
        return jsonify({"error": "Bad Request", "message": "No file was sent."}), 400

    pdf_bytes = upload.read()
    if not pdf_bytes:
        return jsonify({"error": "Bad Request", "message": "The file was empty."}), 400

    try:
        vocabulary = json.loads(request.form.get('vocabulary') or '{}')
    except ValueError:
        return jsonify({"error": "Bad Request",
                        "message": "vocabulary must be JSON."}), 400
    if not isinstance(vocabulary, dict):
        return jsonify({"error": "Bad Request",
                        "message": "vocabulary must be a JSON object."}), 400

    model = (request.form.get('model') or 'claude').strip().lower()

    if model not in extract.SUPPORTED_MODELS:
        return jsonify({
            "error": "Bad Request",
            "message": f"'{model}' cannot read a question paper yet. "
                       "Supported: " + ", ".join(extract.SUPPORTED_MODELS),
        }), 400

    try:
        result = await extract.extract_from_pdf(pdf_bytes, vocabulary, model)
    except rag.PdfEncrypted as exc:
        # Separated from a corrupt file: this one is fixable in a few seconds,
        # and saying "could not be read" would send the teacher looking for
        # another paper instead of removing the password from this one.
        return jsonify({
            "status": "encrypted",
            "message": f"This PDF cannot be opened - {exc}",
            "questions": [],
        }), 200
    except rag.PdfUnreadable as exc:
        # The uploader's problem to fix, so it is a verdict rather than a fault.
        return jsonify({
            "status": "failed",
            "message": f"The PDF could not be read. ({exc})",
            "questions": [],
        }), 200
    except Exception as exc:
        app_logger.exception("Question extraction failed")
        return jsonify({"error": "Extraction Failed", "message": str(exc)}), 500

    app_logger.info("Extracted %s question(s) from %s",
                    len(result["questions"]), upload.filename)
    return jsonify(result), 200


@main_bp.route('/classify_questions', methods=['POST'])
async def classify_questions_endpoint():
    """
    Files questions the teacher already has against their course.

    JSON:
        questions   - the question texts, in the order they were written
        vocabulary  - the lists the classification must choose from, exactly as
                      /extract_questions takes them
        model       - which provider does the work (default "claude")

    The text is returned unchanged; only the labels are the model's.
    """
    from app import extract

    body = request.get_json(silent=True) or {}
    questions = body.get('questions')

    if not isinstance(questions, list):
        return jsonify({"error": "Bad Request",
                        "message": "questions must be a list."}), 400
    if not questions:
        return jsonify({"error": "Bad Request",
                        "message": "No questions were sent."}), 400
    if len(questions) > MAX_QUESTIONS:
        return jsonify({
            "error": "Payload Too Large",
            "message": f"At most {MAX_QUESTIONS} questions per upload. "
                       f"This file has {len(questions)}. Split it and try again.",
        }), 413

    vocabulary = body.get('vocabulary') or {}
    if not isinstance(vocabulary, dict):
        return jsonify({"error": "Bad Request",
                        "message": "vocabulary must be a JSON object."}), 400

    model = (body.get('model') or 'claude').strip().lower()
    if model not in extract.SUPPORTED_MODELS:
        return jsonify({
            "error": "Bad Request",
            "message": f"'{model}' cannot classify questions yet. "
                       "Supported: " + ", ".join(extract.SUPPORTED_MODELS),
        }), 400

    try:
        result = await extract.classify_questions(questions, vocabulary, model)
    except Exception as exc:
        app_logger.exception("Question classification failed")
        return jsonify({"error": "Classification Failed", "message": str(exc)}), 500

    app_logger.info("Classified %s question(s)", len(result["questions"]))
    return jsonify(result), 200
