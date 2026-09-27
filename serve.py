"""Production entry point for Windows, Linux and container hosting."""

import os

from waitress import serve

from app import app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")
    # One process keeps a single copy of the large TensorFlow model in memory.
    serve(app, host="0.0.0.0", port=port, threads=2)
