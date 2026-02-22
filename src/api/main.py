"""FastAPI inference service with Grad-CAM response."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.config import CLASS_NAMES_PATH, IMAGE_SIZE, MODEL_PATH
from src.explainability.gradcam import generate_gradcam, overlay_heatmap
from src.utils.io import load_class_names

app = FastAPI(title="Skin Disease Detection API")


@app.on_event("startup")
def load_artifacts() -> None:
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        raise RuntimeError("Model artifacts not found. Run training first.")
    app.state.model = tf.keras.models.load_model(MODEL_PATH)
    app.state.class_names = load_class_names(CLASS_NAMES_PATH)


def preprocess_pil(image: Image.Image) -> tuple[np.ndarray, tf.Tensor]:
    rgb = np.array(image.convert("RGB").resize(IMAGE_SIZE))
    x = rgb.astype("float32") / 255.0
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    return rgb, tf.expand_dims(x, axis=0)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image.")

    raw = await file.read()
    image = Image.open(BytesIO(raw))
    rgb, x = preprocess_pil(image)

    preds = app.state.model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])

    heatmap = generate_gradcam(app.state.model, x, class_index=idx)
    overlay = overlay_heatmap(rgb, heatmap)

    buffer = BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    heatmap_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "predicted_class": app.state.class_names[idx],
        "confidence": confidence,
        "heatmap_base64": heatmap_b64,
    }
