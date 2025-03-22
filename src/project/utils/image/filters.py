import cv2
import numpy as np


def gaussian_blur(img: np.ndarray, ksize: int = 7) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), 0)  # type: ignore
