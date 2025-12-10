ReSkinAI — All-in-One (Train → Infer → Grad-CAM → App)

AI demo that classifies basic skin conditions from a face photo: acne, dark_spot, redness, normal.
Built with PyTorch EfficientNet-B3 and Grad-CAM. One file does it all: reskin_all_in_one.py.

⚠️ Disclaimer: This is an educational demo, not medical advice. Always consult a dermatologist for diagnosis/treatment.

✨ Features

Single script: organize dataset → train → predict → Grad-CAM → run Streamlit app.

Flexible data intake:

From CSV + raw images (recursively found even if nested).

From an existing class-folder structure.

From a ready train/val split.

Auto train/val split if you supply only class folders.

Grad-CAM heatmap to visualize what the model looked at.

Streamlit app for easy uploads and visualization.

📁 Repo Structure (recommended)
ReSkinAI/
├─ reskin_all_in_one.py          # <— all-in-one pipeline
├─ README.md
├─ data/
│  ├─ raw/                       # (optional) unzip all images here
│  ├─ train/                     # auto-created if needed
│  │  ├─ acne/ ...
│  │  ├─ dark_spot/ ...
│  │  ├─ redness/ ...
│  │  └─ normal/ ...
│  └─ val/
│     ├─ acne/ ...
│     ├─ dark_spot/ ...
│     ├─ redness/ ...
│     └─ normal/ ...
├─ models/
│  └─ efficientnet_b3_best.pt    # saved after training
└─ skin_defects.csv              # (optional) file,label or image,class

🧰 Requirements

Python 3.10+

PyTorch + TorchVision

Pillow, NumPy, Pandas, TQDM, scikit-learn

OpenCV

Grad-CAM (for heatmaps)

Streamlit (for the app)

Install (Windows / macOS / Linux):

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install torch torchvision torchaudio Pillow numpy pandas tqdm scikit-learn opencv-python grad-cam streamlit

📦 Dataset Options
Option A — CSV + raw images

Put all unzipped images anywhere under data/raw/ (nested is fine).

Provide a CSV with at least two columns (names are flexible):

file or image → relative path or filename

label or class → one of:

acne, dark_spot, redness, normal

common variants like pimples, hyperpigmentation, rosacea, healthy are auto-mapped.

Example CSV:

file,label
person_001.jpg,acne
folderA/img002.png,dark_spot
photos/jane.png,redness
abc/xyz.jpg,normal

Option B — Class-folder

Supply a folder that already has:

<your_folder>/
  acne/ *.jpg|*.png...
  dark_spot/ ...
  redness/ ...
  normal/ ...


The script will auto-split into data/train and data/val.

Option C — Ready train/val

If you already have data/train/<class> and data/val/<class>, the script will just use them.

🚆 Train

Run all commands from the folder that contains reskin_all_in_one.py.

A) CSV + raw images
python reskin_all_in_one.py train --data_dir data --csv "./skin_defects.csv" --raw_dir "./data/raw" --val_ratio 0.15 --epochs 10 --bs 16

B) Class-folder (auto-split)
python reskin_all_in_one.py train --data_dir data --class_dir "D:/my_skin_classes" --val_ratio 0.15 --epochs 10 --bs 16

C) Already have train/val
python reskin_all_in_one.py train --data_dir data --epochs 10 --bs 16


Outputs

Saves the best model to models/efficientnet_b3_best.pt

Prints a classification report at the end.

🔎 Inference (single image)
python reskin_all_in_one.py infer --img "./some_face.jpg" --weights "./models/efficientnet_b3_best.pt"


Example output:

Prediction: acne (0.87)
- acne: 0.87
- dark_spot: 0.05
- redness: 0.06
- normal: 0.02

🔥 Grad-CAM Heatmap
python reskin_all_in_one.py cam --img "./some_face.jpg" --weights "./models/efficientnet_b3_best.pt" --out cam_overlay.jpg


This writes cam_overlay.jpg with a heatmap overlay.

🌐 Streamlit App
streamlit run reskin_all_in_one.py -- app --weights "./models/efficientnet_b3_best.pt"


Upload a face photo

See prediction + probabilities + Grad-CAM heatmap (requires grad-cam installed)

⚙️ Training/Model Details

Backbone: EfficientNet-B3 (torchvision)

Image size: 300×300

Augmentations: resize, horizontal flip, color jitter

Optimizer: AdamW (lr=3e-4)

Loss: CrossEntropy

Best-val checkpoint saved automatically

🧪 Quick Sanity Checklist

Do you see non-zero image counts in each class for both train/ and val/?

Are image files actually inside the class folders (not empty)?

On Windows paths with spaces (OneDrive), always wrap paths in quotes "...".

🩹 Troubleshooting

Found no valid file for the classes ...
→ Your folders are empty or images aren’t in supported formats. Confirm with:

# Windows
tree /F data
# macOS/Linux
find data -maxdepth 3 -type f | wc -l


FileNotFoundError: skin_defects.csv
→ CSV isn’t in the current directory. Run dir/ls to verify or pass a full path.

Relative import / module errors
→ Not applicable here (single file). If you split files, run as a module:
python -m src.train and ensure src/__init__.py exists.

num_workers issues on Windows
→ If DataLoader hangs, change num_workers=0 in the code.

Slow training on CPU
→ Works, just slower. For GPU, install the CUDA-enabled PyTorch build.

📄 License

MIT (or your choice). Add a license file if you plan to share/distribute.

🙌 Credits

Built with PyTorch, TorchVision, Grad-CAM, and Streamlit. Inspired by dermatology computer-vision demos.
