# run.py

from asgiref.wsgi import WsgiToAsgi

from app import create_app
from app.config import Config

flask_app = create_app()

app = WsgiToAsgi(flask_app)


# Start the service with:  python run.py
#
# Host, port and worker count come from .env by way of app/config.py, so a
# deployment that needs a different port edits PORT in .env and restarts. The
# Dockerfile and docker-compose read the same value, and nothing else in the
# project hard-codes it.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "run:app",
        host=Config.HOST,
        port=Config.PORT,
        # uvicorn wants None, not 1, when it should stay single-process.
        workers=Config.WORKERS if Config.WORKERS > 1 else None,
    )
