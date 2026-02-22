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

Use direct class folders:

```text
your_dataset/
├── acne/
├── eczema/
├── melanoma/
└── psoriasis/
```

`src/train.py` uses `validation_split` to create train/validation sets at runtime.

## 3) Setup (Linux/macOS / Codespaces)

```bash
# IMPORTANT: run from repository root (where README.md and requirements.txt exist)
pwd
ls

# If you are in GitHub Codespaces, this is usually:
cd /workspaces/ReSkinAI

# Generic alternative:
# cd /path/to/ReSkinAI

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4) Setup (Windows PowerShell)

```powershell
cd C:\path\to\ReSkinAI
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `Activate.ps1` is blocked, run once in PowerShell (as current user):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 5) Train

### Linux/macOS

```bash
# Run this from repo root, not from inside your_dataset/
python -m src.train --data_dir ./your_dataset --epochs 20
```

### Windows PowerShell

```powershell
python -m src.train --data_dir C:\path\to\your_dataset --epochs 20
```

Generated artifacts:
- `artifacts/efficientnet_b3_skin_classifier.keras`
- `artifacts/class_names.json`
- `artifacts/training_history.png`
- `artifacts/confusion_matrix.png`

## 6) Run Inference API (FastAPI)

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Test endpoint (Linux/macOS):

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.jpg"
```

Test endpoint (Windows PowerShell):

```powershell
curl.exe -X POST "http://localhost:8000/predict" -F "file=@C:\path\to\image.jpg"
```

Response JSON:

```json
{
  "predicted_class": "eczema",
  "confidence": 0.9231,
  "heatmap_base64": "<base64-png>"
}
```

## 7) Run Frontend (Streamlit)

```bash
streamlit run src/ui/app.py
```

If your API runs elsewhere, set `api_url` in Streamlit secrets:

```toml
# .streamlit/secrets.toml
api_url = "https://your-api-domain/predict"
```

Features:
- Upload image
- Camera capture
- Prediction and confidence display
- Grad-CAM heatmap visualization

## 8) Troubleshooting

### Linux/macOS / Codespaces

- **`cd /workspace/ReSkinAI: No such file or directory`**: use `/workspaces/ReSkinAI` in Codespaces (note the trailing `s`), or `pwd`/`ls` to find your actual repo path.
- **`ERROR: Could not open requirements file ... requirements.txt`**: you are not in repo root. Run `cd /workspaces/ReSkinAI` (or your repo root), then `ls` and confirm `requirements.txt` is present.
- **`ModuleNotFoundError: No module named 'src'`**: you ran training from the wrong directory (e.g., inside `your_dataset/`). Go back to repo root and run: `python -m src.train --data_dir ./your_dataset --epochs 20`.
- **`your_dataset/` tree text fails with `command not found`**: that block is documentation, not a command. Create folders using: `mkdir -p your_dataset/{acne,eczema,melanoma,psoriasis}`.

### Windows PowerShell

- **`cd /workspace/ReSkinAI` fails**: that path is Linux-only. Use your real Windows path, e.g. `cd C:\Users\lenovo\Desktop\ReSkinAI`.
- **`source` not recognized**: `source` is bash syntax. In PowerShell use `\.venv\Scripts\Activate.ps1`.
- **`Python was not found`**: install Python 3.10+ from python.org and verify:
  - `py --version`
  - `python --version`
- **`pip` not recognized**: use `python -m pip` instead of `pip` directly.

## 9) Production Notes

- Add HTTPS + auth before exposing API publicly.
- Use object storage for uploaded images and observability logs.
- Wrap model loading with startup checks in deployment pipeline.
- Consider mixed precision and fine-tuning (unfreeze top blocks) for better accuracy.
