from __future__ import annotations

import cv2
import numpy as np


def to_greyscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img
