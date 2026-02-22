"""Streamlit frontend for upload/camera prediction and Grad-CAM display."""

from __future__ import annotations

import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

API_URL = st.secrets.get("api_url", "http://localhost:8000/predict")

st.set_page_config(page_title="Skin Disease Detection", page_icon="🩺", layout="centered")
st.title("🩺 AI Skin Disease Detection")
st.write("Upload or capture a skin image to get class prediction with Grad-CAM explainability.")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
captured = st.camera_input("Or capture image")

image_source = captured if captured is not None else uploaded

if image_source is not None:
    image = Image.open(image_source)
    st.image(image, caption="Input image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Running inference..."):
            files = {"file": ("input.png", image_source.getvalue(), "image/png")}
            response = requests.post(API_URL, files=files, timeout=60)

        if response.status_code != 200:
            st.error(f"Request failed: {response.text}")
        else:
            data = response.json()
            st.success(f"Prediction: **{data['predicted_class']}**")
            st.write(f"Confidence: **{data['confidence']:.2%}**")

            heatmap = Image.open(BytesIO(base64.b64decode(data["heatmap_base64"])))
            st.image(heatmap, caption="Grad-CAM", use_column_width=True)
