# run.py

import sys

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
def _worker_count() -> int | None:
    """
    How many workers to run, and why it may not be what .env asked for.

    uvicorn spreads workers by handing a listening socket to forked children.
    Windows has no fork, and the attempt dies with an unexplained
    "WinError 10022: An invalid argument was supplied" - the parent starts, no
    child survives, and nothing serves. The service is deployed on Linux, where
    the setting works; this only stops a developer's machine from failing in a
    way that says nothing about the cause.
    """
    wanted = Config.WORKERS

    if wanted > 1 and sys.platform == "win32":
        print(f"WORKERS={wanted} ignored: uvicorn cannot run multiple workers "
              f"on Windows. Running one. The deployed Linux container will use "
              f"all {wanted}.")
        return None

    # uvicorn wants None, not 1, when it should stay single-process.
    return wanted if wanted > 1 else None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "run:app",
        host=Config.HOST,
        port=Config.PORT,
        workers=_worker_count(),
    )
