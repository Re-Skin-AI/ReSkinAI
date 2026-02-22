"""Training entrypoint for skin disease classifier."""

from __future__ import annotations

import argparse

import numpy as np

from src.config import BATCH_SIZE, CLASS_NAMES_PATH, EPOCHS
from src.data.dataset import dataset_to_numpy, load_datasets
from src.models.efficientnet import build_model
from src.training.callbacks import get_callbacks
from src.training.visualization import plot_confusion_matrix, plot_history
from src.utils.io import save_class_names


def main(data_dir: str, epochs: int) -> None:
    train_ds, val_ds, class_names = load_datasets(data_dir=data_dir, batch_size=BATCH_SIZE)
    model = build_model(num_classes=len(class_names))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(),
    )

    plot_history(history)

    x_val, y_val = dataset_to_numpy(val_ds)
    y_true = np.argmax(y_val, axis=1)
    y_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
    plot_confusion_matrix(y_true, y_pred, class_names)
    save_class_names(CLASS_NAMES_PATH, class_names)

    print("Training finished. Artifacts written to ./artifacts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root with class folders")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    main(data_dir=args.data_dir, epochs=args.epochs)
