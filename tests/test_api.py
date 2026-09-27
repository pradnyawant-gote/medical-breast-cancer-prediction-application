"""HTTP contract checks using generated images, never patient data."""

import io
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import ALLOWED_ORIGINS, app


def sample_png():
    pixels = np.zeros((128, 128, 3), dtype=np.uint8)
    pixels[32:96, 40:88] = 130
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_and_validation(self):
        self.assertEqual(self.client.get("/api/v1/health").json["status"], "ok")
        response = self.client.post("/api/v1/predict", json={"image": "wrong format"})
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json["error"]["code"], "unsupported_media_type")
        response = self.client.post(
            "/api/v1/predict", data={}, content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"]["code"], "missing_image")
        response = self.client.post(
            "/api/v1/predict", data={"image": (io.BytesIO(b"invalid"), "bad.png")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"]["code"], "invalid_image")

        response = self.client.post(
            "/api/v1/predict?include_visuals=maybe",
            data={"image": (sample_png(), "scan.png")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"]["code"], "invalid_option")

        response = self.client.get("/api/v1/predict")
        self.assertEqual(response.status_code, 405)
        self.assertEqual(response.json["error"]["code"], "method_not_allowed")

    def test_inference_failure_is_not_reported_as_bad_image(self):
        with patch("app.predict_scores", side_effect=ValueError("invalid model shape")):
            response = self.client.post(
                "/api/v1/predict", data={"image": (sample_png(), "scan.png")}
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json["error"]["code"], "prediction_failed")

    def test_oversized_request_returns_json(self):
        previous_limit = app.config["MAX_CONTENT_LENGTH"]
        app.config["MAX_CONTENT_LENGTH"] = 512
        try:
            response = self.client.post(
                "/api/v1/predict",
                data={"image": (io.BytesIO(b"x" * 1024), "big.png")},
            )
            self.assertEqual(response.status_code, 413)
            self.assertEqual(response.json["error"]["code"], "image_too_large")
        finally:
            app.config["MAX_CONTENT_LENGTH"] = previous_limit

    def test_prediction_and_optional_visuals(self):
        response = self.client.post(
            "/api/v1/predict", data={"image": (sample_png(), "scan.png")}
        )
        self.assertEqual(response.status_code, 200)
        body = response.json
        self.assertIn(body["predicted_class"], ("benign", "malignant", "normal"))
        self.assertEqual(set(body["scores"]), {"benign", "malignant", "normal"})
        self.assertAlmostEqual(sum(body["scores"].values()), 1, places=4)
        self.assertNotIn("visuals", body)

        response = self.client.post(
            "/api/v1/predict?include_visuals=true",
            data={"image": (sample_png(), "scan.png")},
        )
        self.assertEqual(response.status_code, 200)
        visuals = response.json["visuals"]
        self.assertEqual(visuals["method"], "LayerCAM")
        self.assertTrue(visuals["edge_png_data_url"].startswith("data:image/png;base64,"))
        self.assertTrue(visuals["focus_png_data_url"].startswith("data:image/png;base64,"))

    def test_cors_requires_an_explicit_origin(self):
        allowed = "https://example.test"
        ALLOWED_ORIGINS.add(allowed)
        try:
            response = self.client.options(
                "/api/v1/predict", headers={"Origin": allowed}
            )
            self.assertEqual(response.headers.get("Access-Control-Allow-Origin"), allowed)
            self.assertIn("X-API-Key", response.headers.get("Access-Control-Allow-Headers"))
            response = self.client.options(
                "/api/v1/predict", headers={"Origin": "https://other.test"}
            )
            self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))
        finally:
            ALLOWED_ORIGINS.remove(allowed)

    def test_api_only_mode_hides_website_routes(self):
        with patch("app.API_ONLY", True):
            self.assertEqual(self.client.get("/").json["service"], "SonoLab API")
            self.assertEqual(self.client.get("/api/v1").json["version"], "v1")
            response = self.client.get("/predict")
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json["error"]["code"], "not_found")

    def test_optional_api_key(self):
        with patch("app.API_KEY", "test-secret"):
            response = self.client.post(
                "/api/v1/predict", data={"image": (sample_png(), "scan.png")}
            )
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.json["error"]["code"], "unauthorized")
            with patch("app.predict_scores", return_value=(np.array([0.1, 0.2, 0.7]), None)):
                response = self.client.post(
                    "/api/v1/predict",
                    headers={"X-API-Key": "test-secret"},
                    data={"image": (sample_png(), "scan.png")},
                )
            self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
