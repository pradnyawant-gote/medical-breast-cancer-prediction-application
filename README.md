# SonoLab

A Flask website for the supplied `model3.h5` breast ultrasound classifier.

## Run locally

Use Python 3.11 or 3.12, then run these commands from this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. The navigation has separate Home, Product, Features, Service, and About us pages. Use **Predict now** to open <http://127.0.0.1:5000/predict>. The first prediction can take longer while TensorFlow loads the model. The model file is large, so allow enough memory for it.

The sun/moon button in the header switches between dark and light themes. The selected theme is saved in the browser and applies to every page.

The app accepts PNG and JPEG images up to 10 MB. It resizes each image to 128 × 128 RGB and divides pixel values by 255, matching the notebook's test preprocessing. The model's three output positions are interpreted in Keras's alphabetical class order: `benign`, `malignant`, `normal`.

The app does not keep uploaded images after the request. This is a research demo, not a medical diagnostic tool.

After classification, the prediction page also displays a Sobel edge map and a class-specific LayerCAM focus overlay. For benign or malignant predictions, it outlines a high-influence area only when the map is sufficiently localized; a Normal prediction does not show a contour. `model3.h5` is an image classifier, not a lesion segmentation model. Its 25 × 25 focus map is enlarged for display, so the contour is an explanation of the classifier's response, not a validated lesion boundary. A missing contour or a Normal prediction does not rule out an abnormality. Precise lesion segmentation requires a separately trained segmentation model and validation against expert masks.

The training notebook currently collects every file in the BUSI class folders without excluding `_mask` images. Its saved table contains 1,578 entries, while the [dataset card](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset/metadata) lists 780 original ultrasound images alongside ground-truth images. The supplied classifier should therefore be retrained and evaluated on original scans only before interpreting its scores or focus maps as reliable.

## JSON API

The website and API use the same image preparation and class ordering. Start locally with `python serve.py` after installing the requirements, then use:

```powershell
curl.exe http://127.0.0.1:8000/api/v1/health
curl.exe -F "image=@sample.png" http://127.0.0.1:8000/api/v1/predict
```

`POST /api/v1/predict` accepts a multipart PNG or JPEG in the `image` field. Its response has a lowercase `predicted_class`, `scores` for `benign`, `malignant`, and `normal` as numbers from 0 to 1, the model input size, and a research-use notice. To also return PNG data URLs for the edge map, LayerCAM focus overlay, and optional contour, call `/api/v1/predict?include_visuals=true`. These data URLs can make the JSON response large. The contour is model influence, not a lesion segmentation.

Errors use `{"error":{"code":"...","message":"..."}}`, with 400 for a missing or invalid image, 413 for an oversized request, 415 for a non-multipart request, and 503 when the model file is unavailable. `GET /api/v1/health` checks that the model file exists without loading it.

By default browser calls from another origin are blocked. If the frontend is hosted separately, set `CORS_ORIGINS` to a comma-separated list of exact origins such as `https://example.com`. The website and API on the same host need no CORS setting.

## Deploy the model as an API

Use `Dockerfile.api` when you want to deploy the model without the website. Its `/` and `/api/v1` routes return JSON endpoint information, and its prediction route is the same one documented above. The model loads before the server starts accepting requests.

```sh
git lfs pull
docker build -f Dockerfile.api -t sonolab-api .
docker run --rm -p 8000:8000 sonolab-api
```

Test the deployed API at `http://localhost:8000/api/v1/health` and send a PNG or JPEG with `curl -F "image=@sample.png" http://localhost:8000/api/v1/predict`. For a VPS, run the container with `--restart unless-stopped` and put an HTTPS reverse proxy in front of port 8000. Container platforms can set `PORT` and probe `/api/v1/health`.

For a public API, set `API_KEY` as a secret environment variable. Prediction requests must then include the same value in an `X-API-Key` header; health checks remain public. Without `API_KEY`, the prediction route is open. Browser clients on another origin also need that origin in `CORS_ORIGINS`.

## Deploy the website and API on Render Free

The included `render.yaml` uses Render's **Free** web service plan, `Dockerfile.free`, and `/api/v1/health`. The original TensorFlow container exceeded Free's 512 MB memory limit when measured locally, so this deployment runs an approximately 80 MB float16 LiteRT copy of the same model. One Render service now serves the website at `/` and `/predict` and the JSON API at `/api/v1`, `/api/v1/health`, and `/api/v1/predict`. The website posts images to the server on the same origin, so its visitors do not need the API key. Direct JSON prediction requests still require `X-API-Key` when `API_KEY` is configured.

The Render Free website shows the uploaded image, all three class scores, and a Sobel edge view. It accepts PNG/JPEG uploads up to 10 MB and 8 megapixels to stay within the memory limit. It does not show the TensorFlow LayerCAM focus or contour because the lightweight runtime cannot compute model gradients; `include_visuals=true` on the JSON API returns a 400 error. Edge detection is image processing, not lesion localization. The full TensorFlow website remains available through the regular `Dockerfile` on a host with more memory.

Commit `model_free.tflite` as a regular Git file so Render receives the actual model without depending on Git LFS during its build. The original `model3.h5` remains in Git LFS for the full website/API container. To regenerate and check the smaller model locally, run `python convert_model_free.py` and `python verify_model_free.py` in an environment with TensorFlow installed. The conversion check compares the two models on 10 deterministic inputs; it is not a clinical accuracy test.

The `sonolab-api` Blueprint has already been created. Commit and push these changes to its linked `main` branch; Render will rebuild that same service. Do not create a second Blueprint or web service. After its deploy shows **Live**, open `https://sonolab-api.onrender.com/` for the website, `https://sonolab-api.onrender.com/predict` for the upload page, and `https://sonolab-api.onrender.com/api/v1/health` for the health check.

For direct API calls, send a PNG or JPEG as the multipart field `image` and include `X-API-Key`:

```sh
curl -H "X-API-Key: YOUR_KEY" -F "image=@sample.png" https://sonolab-api.onrender.com/api/v1/predict
```

Free instances have 0.1 CPU and 512 MB RAM, spin down after 15 minutes idle, and may take about a minute to wake. Predictions can therefore be slow. This is a research demo and is not a medical diagnostic service.

## Deployment

The included Dockerfile runs the Flask application with Waitress, a production WSGI server. The same `serve.py` entry point runs on Windows or Linux. It listens on the `PORT` environment variable (default 8000) and uses one process so the large model is loaded once.

```sh
git lfs pull
docker build -t sonolab .
docker run --rm -p 8000:8000 sonolab
```

On a VPS, replace the temporary `docker run` command with `docker run -d --name sonolab --restart unless-stopped -p 8000:8000 sonolab`, and put an HTTPS reverse proxy in front of it. For a container hosting platform, point it at the Dockerfile, expose its assigned `PORT`, and use `/api/v1/health` as the health check. Ensure Git LFS has downloaded the real `model3.h5` before building; the file is about 483 MB, so the host also needs memory for TensorFlow and model inference. The production entry point loads the model before accepting requests. The Flask development server in `app.py` is only for local development.

## Visual design

The navy, orange, cyan, and oversized-type direction was inspired by the [MLify Dribbble reference](https://dribbble.com/shots/24892774--MLify-AI-Machine-Learning-Algorithms-Header-Website) supplied for this project. The hero artwork is an original asset at `static/images/hero-machine.png`, made with the built-in image generation tool. The final prompt was:

> Use case: stylized-concept. Asset type: transparent-background hero illustration for a breast ultrasound machine-learning website. Create an original premium 3D isometric computation device: a dark graphite and brushed-metal processing console with subtle cyan pixel lights and a warm orange illuminated edge; above it, a sculptural silver data funnel with a thin glowing cyan orbital ring and a few small floating abstract geometric data glyphs (shapes only, no letters). Beneath it, a restrained translucent gridded scanning plane and a couple of small metallic modules. Three-quarter front view, centered composition, clear silhouette, highly polished art direction like a high-end technology product render. Palette midnight graphite, gunmetal silver, vivid orange and electric cyan. Dramatic but precise studio lighting, realistic depth and material texture. Transparent background with genuine alpha; no floor or backdrop. No words, no labels, no logo, no watermark, no UI cards, no breast anatomy. Intended to sit over a nearly black navy website hero.
