# ReSkinAI - EfficientNet-B3 Skin Disease Detection

Production-ready end-to-end project for **AI-based skin disease detection** using **TensorFlow/Keras EfficientNet-B3** with **Grad-CAM explainability**, a **FastAPI inference backend**, and a **Streamlit frontend**.

## 1) Project Structure

```text
ReSkinAI/
├── artifacts/                         # Saved models and plots (generated)
├── dataset/
│   ├── sample/
│   │   ├── train/
│   │   │   ├── acne/
│   │   │   ├── eczema/
│   │   │   ├── melanoma/
│   │   │   └── psoriasis/
│   │   └── val/
│   │       ├── acne/
│   │       ├── eczema/
│   │       ├── melanoma/
│   │       └── psoriasis/
├── src/
│   ├── api/
│   │   └── main.py                    # FastAPI app
│   ├── data/
│   │   └── dataset.py                 # cleaning/loading/preprocessing/augmentation
│   ├── explainability/
│   │   └── gradcam.py                 # Grad-CAM implementation
│   ├── models/
│   │   └── efficientnet.py            # EfficientNet-B3 transfer learning model
│   ├── training/
│   │   ├── callbacks.py               # checkpoint, early stopping, lr scheduler
│   │   └── visualization.py           # history + confusion matrix plots
│   ├── ui/
│   │   └── app.py                     # Streamlit interface
│   ├── utils/
│   │   └── io.py                      # class names and image helpers
│   ├── config.py                      # central config values
│   └── train.py                       # training entrypoint
├── requirements.txt
└── README.md
```

## 2) Dataset Folder Format

Use either direct class folders (recommended):

```text
your_dataset/
├── acne/
├── eczema/
├── melanoma/
└── psoriasis/
```

`src/train.py` automatically creates train/validation split via `validation_split`.

## 3) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4) Train

```bash
python -m src.train --data_dir /path/to/your_dataset --epochs 20
```

Generated artifacts:
- `artifacts/efficientnet_b3_skin_classifier.keras`
- `artifacts/class_names.json`
- `artifacts/training_history.png`
- `artifacts/confusion_matrix.png`

## 5) Run Inference API (FastAPI)

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Test endpoint:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.jpg"
```

Response JSON:

```json
{
  "predicted_class": "eczema",
  "confidence": 0.9231,
  "heatmap_base64": "<base64-png>"
}
```

## 6) Run Frontend (Streamlit)

```bash
streamlit run src/ui/app.py
```

Features:
- Upload image
- Camera capture
- Prediction and confidence display
- Grad-CAM heatmap visualization

## 7) Production Notes

- Add HTTPS + auth before exposing API publicly.
- Use object storage for uploaded images and observability logs.
- Wrap model loading with startup checks in deployment pipeline.
- Consider mixed precision and fine-tuning (unfreeze top blocks) for better accuracy.

