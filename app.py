"""Flask interface for the ultrasound classifier saved in model3.h5."""

import base64
import hmac
import io
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageFilter, UnidentifiedImageError
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model3.h5"
CLASS_NAMES = ("Benign", "Malignant", "Normal")
IMAGE_SIZE = (128, 128)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

Image.MAX_IMAGE_PIXELS = 20_000_000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
ALLOWED_ORIGINS = {
    origin.strip().rstrip("/")
    for origin in os.environ.get("CORS_ORIGINS", "").split(",")
    if origin.strip()
}
API_ONLY = os.environ.get("API_ONLY", "").lower() in {"1", "true", "yes"}
API_KEY = os.environ.get("API_KEY", "")


@app.after_request
def add_api_cors(response):
    """Allow only explicitly configured browser origins to call the JSON API."""
    origin = request.headers.get("Origin", "").rstrip("/")
    if request.path.startswith("/api/") and origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key"
        response.headers.add("Vary", "Origin")
    return response


@lru_cache(maxsize=1)
def get_model():
    # Loading on the first prediction keeps the home page responsive.
    from tensorflow.keras.models import load_model

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)
    if model.input_shape != (None, 128, 128, 3) or model.output_shape[-1] != 3:
        raise ValueError("model3.h5 has an unexpected input or output shape")
    return model


@lru_cache(maxsize=1)
def get_explanation_model():
    """Split the network at a useful spatial feature map for LayerCAM."""
    from tensorflow.keras import Model

    model = get_model()
    convolution_layers = [
        layer for layer in model.layers
        if layer.__class__.__name__ == "Conv2D" and min(layer.output.shape[1:3]) >= 12
    ]
    if not convolution_layers:
        raise RuntimeError("The classifier has no suitable spatial feature map")
    focus_layer = convolution_layers[-1]
    feature_model = Model(inputs=model.inputs, outputs=focus_layer.output)
    return feature_model, model.layers[model.layers.index(focus_layer) + 1:]


def png_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def prepare_image(upload):
    """Apply the same RGB, 128x128 and 1/255 preprocessing as the notebook."""
    try:
        with Image.open(upload.stream) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Please upload a PNG or JPEG image.")
            if image.width * image.height > Image.MAX_IMAGE_PIXELS:
                raise ValueError("This image is too large to process.")
            image = image.convert("RGB")
            resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32) / 255.0

            preview = image.copy()
            preview.thumbnail((520, 390))
            return np.expand_dims(array, axis=0), preview
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("That file could not be read as an image. Try a PNG or JPEG.") from exc


def classify_with_focus(batch):
    """Return class scores and a class-specific LayerCAM activation map."""
    import tensorflow as tf

    feature_model, remaining_layers = get_explanation_model()
    with tf.GradientTape() as tape:
        feature_maps = feature_model([batch], training=False)
        tape.watch(feature_maps)
        dense_input = feature_maps
        for layer in remaining_layers[:-1]:
            dense_input = layer(dense_input, training=False) if layer.__class__.__name__ == "BatchNormalization" else layer(dense_input)
        final_layer = remaining_layers[-1]
        scores = final_layer(dense_input)
        class_index = tf.argmax(scores[0])
        # Use the pre-softmax logit so a saturated score still has a useful gradient.
        class_weights = tf.gather(final_layer.kernel, class_index, axis=1)
        class_logit = tf.reduce_sum(dense_input[0] * class_weights)

    gradients = tape.gradient(class_logit, feature_maps)
    if gradients is None:
        raise RuntimeError("Could not calculate the model focus map")
    # LayerCAM weights each spatial location by its own positive class gradient.
    # The selected layer has a 25x25 map in model3.h5; this remains an explanation,
    # not a pixel-level lesion prediction.
    focus = tf.reduce_sum(feature_maps[0] * tf.nn.relu(gradients[0]), axis=-1)
    focus = tf.maximum(focus, 0)
    focus = focus / (tf.reduce_max(focus) + 1e-8)
    return np.asarray(scores.numpy()), np.asarray(focus.numpy())


def analysis_images(preview, focus, show_region):
    """Make Sobel edges, a LayerCAM overlay and a high-focus contour."""
    gray = np.asarray(preview.convert("L").filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
    padded = np.pad(gray, 1, mode="edge")
    gx = (
        padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:]
        - padded[:-2, :-2] - 2 * padded[1:-1, :-2] - padded[2:, :-2]
    )
    gy = (
        padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]
        - padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - padded[:-2, 2:]
    )
    edge_strength = np.hypot(gx, gy)
    low, high = np.percentile(edge_strength, [70, 98])
    edge_strength = np.clip((edge_strength - low) / max(high - low, 1), 0, 1)
    edge_base = gray[..., None] * np.array([0.12, 0.19, 0.27], dtype=np.float32)
    edge_rgb = np.clip(edge_base + edge_strength[..., None] * [35, 210, 228], 0, 255).astype(np.uint8)

    focus_image = Image.fromarray(np.uint8(np.clip(focus, 0, 1) * 255))
    focus_image = focus_image.resize(preview.size, Image.Resampling.BILINEAR)
    focus_image = focus_image.filter(ImageFilter.GaussianBlur(3))
    focus_values = np.asarray(focus_image, dtype=np.float32) / 255.0
    source = np.asarray(preview, dtype=np.float32)
    orange = np.array([255, 112, 30], dtype=np.float32)
    alpha = (focus_values ** 1.4 * 0.68)[..., None]
    overlay = np.uint8(np.clip(source * (1 - alpha) + orange * alpha, 0, 255))

    region_url = None
    region_reason = None
    if show_region and focus_values.max() > 0:
        mask_image = Image.fromarray(np.uint8(focus_values >= 0.65 * focus_values.max()) * 255)
        mask = np.asarray(mask_image.filter(ImageFilter.MedianFilter(5))) > 0
        coverage = float(mask.mean())
        if 0.003 <= coverage <= 0.35:
            # The contour represents high model influence, not a trained lesion mask.
            region = source.copy()
            region[mask] = source[mask] * 0.78 + orange * 0.22
            padded_mask = np.pad(mask, 1, constant_values=False)
            interior = (
                padded_mask[1:-1, 1:-1]
                & padded_mask[:-2, 1:-1] & padded_mask[2:, 1:-1]
                & padded_mask[1:-1, :-2] & padded_mask[1:-1, 2:]
            )
            border = mask & ~interior
            border = np.asarray(
                Image.fromarray(np.uint8(border) * 255).filter(ImageFilter.MaxFilter(3))
            ) > 0
            region[border] = [255, 177, 73]
            region_url = png_data_url(Image.fromarray(np.uint8(np.clip(region, 0, 255))))
        else:
            region_reason = "The model focus was too small or too diffuse for a useful contour."
    elif show_region:
        region_reason = "The model produced no positive class focus for this image."

    return {
        "edges": png_data_url(Image.fromarray(edge_rgb)),
        "focus": png_data_url(Image.fromarray(overlay)),
        "region": region_url,
        "focus_available": bool(np.max(focus) > 1e-6),
        "focus_resolution": f"{focus.shape[1]} × {focus.shape[0]}",
        "region_reason": region_reason,
    }


def predict_scores(batch, include_focus=False):
    """Run the shared classifier and validate its three output probabilities."""
    if include_focus:
        scores, focus = classify_with_focus(batch)
    else:
        scores = np.asarray(get_model().predict(batch, verbose=0))
        focus = None
    if scores.shape != (1, len(CLASS_NAMES)) or not np.all(np.isfinite(scores)):
        raise RuntimeError("The model returned an invalid prediction")
    probabilities = scores[0]
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=0.01):
        raise RuntimeError("The model returned invalid class probabilities")
    return probabilities, focus


def api_error(code, message, status):
    return jsonify({"error": {"code": code, "message": message}}), status


def api_description():
    return {
        "service": "SonoLab API",
        "version": "v1",
        "endpoints": {
            "health": "GET /api/v1/health",
            "predict": "POST /api/v1/predict",
        },
        "upload": "Send a PNG or JPEG as multipart/form-data in the image field.",
        "notice": "Research use only. This is not a medical diagnosis.",
    }


@app.before_request
def api_deployment_mode():
    if API_ONLY:
        if request.path == "/":
            return jsonify(api_description())
        if not request.path.startswith("/api/"):
            return api_error("not_found", "This deployment serves only the JSON API.", 404)
    if API_KEY and request.path == "/api/v1/predict" and request.method == "POST":
        supplied_key = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(supplied_key, API_KEY):
            return api_error("unauthorized", "A valid X-API-Key header is required.", 401)


@app.get("/api/v1")
def api_index():
    return jsonify(api_description())


@app.get("/api/v1/health")
def api_health():
    """Lightweight check for hosting health probes; it does not load the model."""
    if not MODEL_PATH.is_file():
        return api_error("model_unavailable", "The model file is missing.", 503)
    return jsonify({"status": "ok", "model": MODEL_PATH.name})


@app.post("/api/v1/predict")
def api_predict():
    """Predict from a multipart PNG/JPEG field named `image`."""
    if request.mimetype != "multipart/form-data":
        return api_error("unsupported_media_type", "Send multipart/form-data with an image field.", 415)
    visuals_option = request.args.get("include_visuals", "false").lower()
    if visuals_option not in {"true", "false", "1", "0"}:
        return api_error("invalid_option", "include_visuals must be true or false.", 400)
    include_visuals = visuals_option in {"true", "1"}
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return api_error("missing_image", "Add a PNG or JPEG file in the image field.", 400)

    try:
        batch, preview = prepare_image(upload)
    except ValueError as exc:
        return api_error("invalid_image", str(exc), 400)

    try:
        probabilities, focus = predict_scores(batch, include_focus=include_visuals)
        best_index = int(np.argmax(probabilities))
        response = {
            "model": MODEL_PATH.name,
            "predicted_class": CLASS_NAMES[best_index].lower(),
            "scores": {
                name.lower(): round(float(value), 6)
                for name, value in zip(CLASS_NAMES, probabilities)
            },
            "input_size": {"width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1]},
            "notice": "Research use only. This is not a medical diagnosis.",
        }
        if include_visuals:
            visuals = analysis_images(preview, focus, show_region=best_index != 2)
            response["visuals"] = {
                "method": "LayerCAM",
                "focus_resolution": visuals["focus_resolution"],
                "edge_png_data_url": visuals["edges"],
                "focus_png_data_url": visuals["focus"],
                "contour_png_data_url": visuals["region"],
                "contour_reason": visuals["region_reason"] or (
                    "No contour is shown for a normal prediction." if best_index == 2 else None
                ),
                "notice": "The contour shows model influence, not a lesion boundary.",
            }
        return jsonify(response)
    except FileNotFoundError:
        app.logger.exception("Model file is missing")
        return api_error("model_unavailable", "The model is unavailable.", 503)
    except Exception:
        app.logger.exception("API prediction failed")
        return api_error("prediction_failed", "The model could not process this image.", 500)


@app.route("/")
def home():
    return render_template("index.html", page_name="home")


@app.route("/product")
def product():
    return render_template("product.html", page_name="product")


@app.route("/features")
def features():
    return render_template("features.html", page_name="features")


@app.route("/service")
def service():
    return render_template("service.html", page_name="service")


@app.route("/about")
def about():
    return render_template("about.html", page_name="about")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", page_name="predict")

    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return render_template("predict.html", page_name="predict", error="Choose an ultrasound image first."), 400

    try:
        batch, preview = prepare_image(upload)
        probabilities, focus = predict_scores(batch, include_focus=True)
        results = [
            {"name": name, "percent": round(float(probability) * 100, 1)}
            for name, probability in zip(CLASS_NAMES, probabilities)
        ]
        best = results[int(np.argmax(probabilities))]
        visuals = analysis_images(preview, focus, show_region=best["name"] != "Normal")
        return render_template(
            "predict.html", page_name="predict", result=best, results=results,
            preview_url=png_data_url(preview), visuals=visuals,
        )
    except ValueError as exc:
        return render_template("predict.html", page_name="predict", error=str(exc)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return render_template(
            "predict.html", page_name="predict", error="The model could not process this image. Please try again later."
        ), 500


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    if request.path.startswith("/api/"):
        return api_error("image_too_large", "The request must be smaller than 10 MB.", 413)
    return render_template("predict.html", page_name="predict", error="The image must be smaller than 10 MB."), 413


@app.errorhandler(HTTPException)
def http_error(error):
    if request.path.startswith("/api/"):
        return api_error(error.name.lower().replace(" ", "_"), error.description, error.code)
    return error


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=False)
