Absolutely ✅ — here’s your **ready-to-copy README.md** (you can directly paste this into your `README.md` file on GitHub or VS Code).

---

```markdown
# 🧴 ReSkinAI — All-in-One (Train → Infer → Grad-CAM → App)

An AI demo that classifies basic skin conditions from a face photo: **acne**, **dark_spot**, **redness**, and **normal**.  
Built using **PyTorch EfficientNet-B3** and **Grad-CAM** — all in a single file: `reskin_all_in_one.py`.

> ⚠️ **Disclaimer:** This is an educational demo and **not medical advice**.  
> Always consult a certified dermatologist for any diagnosis or treatment.

---

## ✨ Features

- 🚀 **Single Python file** for full pipeline  
  (Organize dataset → Train → Predict → Visualize → Run Streamlit app)
- 📁 **Flexible dataset input**
  - CSV + Raw images
  - Folder with class subdirectories
  - Pre-split train/val folders
- ⚙️ **Auto train/val split** from unsplit datasets
- 🔥 **Grad-CAM visualization** support
- 🌐 **Streamlit web app** for interactive predictions



## 🧰 Setup

### 1️⃣ Create Environment & Install Dependencies
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install torch torchvision torchaudio Pillow numpy pandas tqdm scikit-learn opencv-python grad-cam streamlit
````

---

## 📦 Dataset Options

### Option A — CSV + Raw Images

Put your unzipped images in `data/raw/` and create a CSV file like:

```csv
file,label
person_001.jpg,acne
folderA/img002.png,dark_spot
photos/jane.png,redness
abc/xyz.jpg,normal
```

Supported variants like `pimples`, `blackspot`, `rosacea`, and `healthy` are automatically mapped to the correct class.

### Option B — Class Folder Structure

Organize your dataset as:

```
dataset/
├─ acne/
├─ dark_spot/
├─ redness/
└─ normal/
```

The script will automatically create `data/train` and `data/val` splits.

### Option C — Ready Split (train/val)

If you already have `data/train` and `data/val` with class folders, you’re good to go!

---

## 🚆 Training

### A) CSV + Raw Images

```bash
python reskin_all_in_one.py train --data_dir data --csv "./skin_defects.csv" --raw_dir "./data/raw" --val_ratio 0.15 --epochs 10 --bs 16
```

### B) Class Folder (Auto-Split)

```bash
python reskin_all_in_one.py train --data_dir data --class_dir "./dataset" --val_ratio 0.15 --epochs 10 --bs 16
```

### C) Pre-Split (train/val exists)

```bash
python reskin_all_in_one.py train --data_dir data --epochs 10 --bs 16
```

✅ The best model will be saved automatically to:

```
models/efficientnet_b3_best.pt
```

---

## 🔎 Inference (Single Image Prediction)

Use your trained model to predict on a single image:

```bash
python reskin_all_in_one.py infer --img "./sample.jpg" --weights "./models/efficientnet_b3_best.pt"
```

Example output:

```
Prediction: acne (0.87)
- acne: 0.87
- dark_spot: 0.05
- redness: 0.06
- normal: 0.02
```

---

## 🔥 Grad-CAM Visualization

Generate a heatmap showing where the model focused:

```bash
python reskin_all_in_one.py cam --img "./sample.jpg" --weights "./models/efficientnet_b3_best.pt" --out cam_overlay.jpg
```

Output:
✅ `cam_overlay.jpg` — input image + attention heatmap overlay.

---

## 🌐 Streamlit Web App

Launch the ReSkinAI app in your browser:

```bash
streamlit run reskin_all_in_one.py -- app --weights "./models/efficientnet_b3_best.pt"
```

**Features:**

* Upload any face photo
* Get prediction + probability for each class
* Visual Grad-CAM attention heatmap

---

## ⚙️ Model Details

| Parameter     | Value             |
| ------------- | ----------------- |
| Architecture  | EfficientNet-B3   |
| Input size    | 300×300           |
| Loss          | CrossEntropyLoss  |
| Optimizer     | AdamW             |
| Learning rate | 3e-4              |
| Batch size    | 16                |
| Epochs        | 10 (configurable) |

---

## 🧪 Quick Sanity Checklist

✅ `data/train` and `data/val` exist
✅ Each class folder contains valid `.jpg`, `.png`, or `.webp` files
✅ You’re running the script **from the same directory** as `reskin_all_in_one.py`
✅ For Windows paths with spaces (e.g., OneDrive), always use quotes: `"path with spaces"`

---

## 🩹 Common Errors & Fixes

| Error                                 | Cause                           | Fix                                                   |
| ------------------------------------- | ------------------------------- | ----------------------------------------------------- |
| `Found no valid file for the classes` | Empty or wrong folder structure | Ensure images are inside correct class folders        |
| `FileNotFoundError: skin_defects.csv` | Wrong CSV path                  | Use full path, e.g. `"C:/Users/.../skin_defects.csv"` |
| `relative import`                     | Not applicable now              | This version uses absolute imports only               |
| `DataLoader hang` on Windows          | Worker issue                    | Change `num_workers=0` in DataLoader                  |

---

## 🧠 Example Commands Recap

```bash
# train (csv)
python reskin_all_in_one.py train --data_dir data --csv "./skin_defects.csv" --raw_dir "./data/raw"

# predict
python reskin_all_in_one.py infer --img "./front.jpg" --weights "./models/efficientnet_b3_best.pt"

# gradcam
python reskin_all_in_one.py cam --img "./front.jpg" --weights "./models/efficientnet_b3_best.pt"

# run app
streamlit run reskin_all_in_one.py -- app --weights "./models/efficientnet_b3_best.pt"
```

---

## 🧴 Tech Stack

* **Language:** Python 3.10+
* **Frameworks:** PyTorch, TorchVision
* **Visualization:** Grad-CAM, Streamlit
* **Libraries:** NumPy, Pandas, scikit-learn, OpenCV, TQDM

---

## 📄 License

MIT License © 2025


