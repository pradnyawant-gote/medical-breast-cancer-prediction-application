"""Small API-only server for Render's 512 MB Free instance."""

import os
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
from flask import request
from waitress import serve

import app as webapp


MODEL_PATH = Path(__file__).resolve().parent / "model_free.tflite"
webapp.MODEL_PATH = MODEL_PATH
webapp.API_ONLY = True


class LiteModel:
    """Adapt a single LiteRT interpreter to the existing JSON API contract."""

    def __init__(self, path):
        from ai_edge_litert.interpreter import Interpreter

        self.lock = Lock()
        self.interpreter = Interpreter(model_path=str(path), num_threads=1)
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected one model input and one model output")
        self.input = inputs[0]
        self.output = outputs[0]
        if (tuple(self.input["shape"]) != (1, 128, 128, 3)
                or tuple(self.output["shape"]) != (1, 3)
                or self.input["dtype"] != np.float32
                or self.output["dtype"] != np.float32):
            raise ValueError("Unexpected LiteRT model shape or dtype")

    def predict(self, batch, verbose=0):
        if batch.shape != (1, 128, 128, 3):
            raise ValueError("Unexpected image batch shape")
        with self.lock:
            self.interpreter.set_tensor(self.input["index"], batch)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output["index"])


@lru_cache(maxsize=1)
def get_lite_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)
    return LiteModel(MODEL_PATH)


webapp.get_model = get_lite_model


@webapp.app.before_request
def reject_unsupported_visuals():
    if request.path == "/api/v1/predict" and request.method == "POST":
        if request.args.get("include_visuals", "false").lower() in {"true", "1"}:
            return webapp.api_error(
                "visuals_unavailable",
                "Focus and contour visuals are unavailable on the Free API deployment.",
                400,
            )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")
    get_lite_model()
    # One request at a time also bounds memory while large images are decoded.
    serve(webapp.app, host="0.0.0.0", port=port, threads=1)
