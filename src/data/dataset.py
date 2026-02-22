"""Dataset loading, cleaning, preprocessing and augmentation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from src.config import IMAGE_SIZE, SEED, VAL_SPLIT

AUTOTUNE = tf.data.AUTOTUNE


def _is_valid_image(path: tf.Tensor) -> tf.Tensor:
    """Return bool tensor indicating whether the image can be decoded."""
    raw = tf.io.read_file(path)
    decoded = tf.io.decode_image(raw, channels=3, expand_animations=False)
    return tf.logical_and(tf.shape(decoded)[0] > 0, tf.shape(decoded)[1] > 0)


def clean_dataset(data_dir: str) -> None:
    """Remove unreadable images from class folders in-place."""
    root = Path(data_dir)
    removed = 0
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        for image_path in class_dir.iterdir():
            if not image_path.is_file():
                continue
            try:
                _ = tf.io.decode_image(tf.io.read_file(str(image_path)), channels=3)
            except tf.errors.InvalidArgumentError:
                image_path.unlink(missing_ok=True)
                removed += 1
    print(f"[clean_dataset] Removed {removed} invalid image(s).")


def _augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply random augmentation for the training dataset."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    angle = tf.random.uniform([], minval=-0.15, maxval=0.15)
    image = tfa_image_rotate(image, angle)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def tfa_image_rotate(image: tf.Tensor, angle_rad: tf.Tensor) -> tf.Tensor:
    """Rotate image using projective transform without extra dependency."""
    return tf.keras.layers.RandomRotation(factor=0.15, fill_mode="nearest")(tf.expand_dims(image, 0))[0]


def _normalize(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Scale image to [0,1] and apply EfficientNet preprocessing."""
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label


def load_datasets(data_dir: str, batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Load train and validation datasets from class-folder dataset."""
    clean_dataset(data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )

    class_names = train_ds.class_names

    train_ds = train_ds.map(_normalize, num_parallel_calls=AUTOTUNE).map(_augment, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(_normalize, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def dataset_to_numpy(ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Convert batched dataset to numpy arrays for metrics plots."""
    xs, ys = [], []
    for x_batch, y_batch in ds:
        xs.append(x_batch.numpy())
        ys.append(y_batch.numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
