"""Grad-CAM generation and overlay helpers."""

from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Find last convolutional layer name in model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name
    raise ValueError("No Conv2D layer found for Grad-CAM.")


def generate_gradcam(model: tf.keras.Model, image_tensor: tf.Tensor, class_index: int | None = None) -> np.ndarray:
    """Generate Grad-CAM heatmap from preprocessed image tensor (1,H,W,3)."""
    last_conv_name = find_last_conv_layer(model)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original RGB image."""
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(heatmap_colored, alpha, original_bgr, 1 - alpha, 0)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
