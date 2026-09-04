"""The textbook index: putting a book in, keeping it current, reporting on it.

The portal posts the PDF here when a textbook is uploaded, and gets back
whether it can actually be quoted in generated questions. The file is sent as
bytes rather than as a link on purpose: this service then never needs the
portal's address and never reaches back into it.
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from app import retrieval, usage
from app.config import Config

bp = Blueprint("textbooks", __name__)

app_logger = logging.getLogger("app")
# --- Textbook indexing -----------------------------------------------------
#
# The portal posts the PDF here when a textbook is uploaded, and gets back
# whether it can actually be used for question generation.
#
# The file is sent as bytes rather than as a link on purpose: this service then
# never needs the portal's address, never reaches back into it, and does not
# depend on those uploads staying publicly readable.


@bp.route('/rag/index', methods=['POST'])
def rag_index_endpoint():
    """
    multipart/form-data:
        file      - the PDF                        (required)
        doc_id    - e.g. "1_734"                   (required)
        model     - embedding model (default: the server's RAG_EMBED_MODEL)
        BookName  - title, for citations           (optional)
        BookType  - "Textbook" / "Reference"       (optional)
    """

    if not Config.RAG_ENABLED:
        return jsonify({
            "status": "disabled",
            "message": "Textbook indexing is not enabled on this server.",
        }), 503

    upload = request.files.get('file')
    if upload is None or not upload.filename:
        return jsonify({"error": "Bad Request", "message": "No file was sent."}), 400

    doc_id = (request.form.get('doc_id') or '').strip()
    parsed = retrieval.parse_doc_id(doc_id or upload.filename)
    if not parsed:
        return jsonify({
            "error": "Bad Request",
            "message": "doc_id must look like '1_734' (slot_paper).",
        }), 400
    doc_id, paper_id, slot_no = parsed

    pdf_bytes = upload.read()
    if not pdf_bytes:
        return jsonify({"error": "Bad Request", "message": "The file was empty."}), 400
    # Which embedding model indexes the book. Vectors from different models are
    # not comparable, so this is recorded against the document and a change to
    # it forces a re-index rather than silently mixing them.
    embed_model = (request.form.get('model') or '').strip()

    try:
        with usage.collect() as spent:
            result = retrieval.index_pdf_bytes(
                pdf_bytes, doc_id, paper_id, slot_no,
                request.form.get('BookName', ''), request.form.get('BookType', ''),
                embed_model or None,
            )
        result["usage"] = spent.as_dict()
    except Exception as exc:
        app_logger.exception("RAG index failed for %s", doc_id)
        return jsonify({
            "error": "Indexing Failed",
            "message": str(exc),
            "doc_id": doc_id,
        }), 500

    app_logger.info("RAG indexed %s -> %s", doc_id, result.get("status"))
    return jsonify(result), 200




@bp.route('/rag/prune', methods=['POST'])
def rag_prune_endpoint():
    """
    Makes the index match a course's current book list.

    JSON: {"paper_id": 734,
           "books": [{"doc_id": "1_734", "BookName": ..., "BookType": ...}, ...]}

    The portal sends its whole list on every save; anything indexed for that
    course and not named here is dropped. An empty list is legitimate - it means
    the course has no books left.
    """

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
        result = retrieval.prune_documents(paper_id, books)
    except Exception as exc:
        app_logger.exception("RAG prune failed for paper %s", paper_id)
        return jsonify({"error": "Prune Failed", "message": str(exc)}), 500

    return jsonify(result), 200




@bp.route('/rag/status/<int:paper_id>', methods=['GET'])
def rag_status_endpoint(paper_id: int):
    """Index status of every book of one course, so the portal can show it."""

    if not Config.RAG_ENABLED:
        return jsonify({"status": "disabled", "books": []}), 503

    try:
        return jsonify({"paper_id": paper_id, "books": retrieval.document_status(paper_id)}), 200
    except Exception as exc:
        app_logger.exception("RAG status failed for paper %s", paper_id)
        return jsonify({"error": "Status Unavailable", "message": str(exc)}), 500
