"""I/O helpers for model metadata and image encoding."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np


def save_class_names(path: Path, class_names: list[str]) -> None:
    path.write_text(json.dumps(class_names, indent=2))


def load_class_names(path: Path) -> list[str]:
    return json.loads(path.read_text())


def image_to_base64(image_rgb: np.ndarray) -> str:
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buffer).decode("utf-8")
