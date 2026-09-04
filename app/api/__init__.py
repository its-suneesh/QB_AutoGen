"""The HTTP layer: parse the request, delegate, shape the reply.

One module per feature, each with its own blueprint. Nothing here decides
anything - the work lives in app/services and app/retrieval - so a route stays
short enough to read in one go and the logic under it can be tested without a
web server.
"""

from flask import Flask

from . import extraction, generation, meta, textbooks

# meta last: it owns the catch-all route, which must be registered after every
# real one or it would answer for them.
_BLUEPRINTS = (generation.bp, extraction.bp, textbooks.bp, meta.bp)


def register(app: Flask) -> None:
    for blueprint in _BLUEPRINTS:
        app.register_blueprint(blueprint)
