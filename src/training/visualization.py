"""Plots for training history and confusion matrix."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import CONFUSION_MATRIX_PATH, HISTORY_PLOT_PATH


def plot_history(history) -> None:
    """Save training-vs-validation metrics plot."""
    hist = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(hist["accuracy"], label="train")
    axes[0].plot(hist["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(hist["loss"], label="train")
    axes[1].plot(hist["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(HISTORY_PLOT_PATH, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> None:
    """Create and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Validation Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close(fig)
