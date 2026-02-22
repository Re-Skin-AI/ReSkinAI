"""Global configuration for training and inference."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (300, 300)
BATCH_SIZE = 16
SEED = 42
VAL_SPLIT = 0.2
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.3
EPOCHS = 20

MODEL_PATH = ARTIFACTS_DIR / "efficientnet_b3_skin_classifier.keras"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
HISTORY_PLOT_PATH = ARTIFACTS_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"
