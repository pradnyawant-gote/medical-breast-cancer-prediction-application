"""Production entry point for Windows, Linux and container hosting."""

import os

from waitress import serve

from app import app, get_model


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")
    # Fail startup if the model is missing or unreadable; one process keeps one copy.
    get_model()
    serve(app, host="0.0.0.0", port=port, threads=2)
