"""Flask interface for the ultrasound classifier saved in model3.h5."""

import base64
import io
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from PIL import Image, ImageFilter, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model3.h5"
CLASS_NAMES = ("Benign", "Malignant", "Normal")
IMAGE_SIZE = (128, 128)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

Image.MAX_IMAGE_PIXELS = 20_000_000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


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
        scores, focus = classify_with_focus(batch)
        if scores.shape != (1, len(CLASS_NAMES)) or not np.all(np.isfinite(scores)):
            raise RuntimeError("The model returned an invalid prediction")
        probabilities = scores[0]
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=0.01):
            raise RuntimeError("The model returned invalid class probabilities")
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
    return render_template("predict.html", page_name="predict", error="The image must be smaller than 10 MB."), 413


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=False)
