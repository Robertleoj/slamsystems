import cv2
import numpy as np

from project.utils.colors import Color


def draw_pixels(
    img: np.ndarray, pixels: np.ndarray, color: Color, radius: int = 5, inplace: bool = False
) -> np.ndarray:
    """Draw pixels.

    Note: modifies image in-place. Send in a copy if you don't want your image modified.
    """
    if not inplace:
        img = img.copy()

    for pixel in pixels:
        cv2.circle(img, center=pixel.astype(int), radius=radius, color=color, thickness=-1)

    return img


def draw_lines(
    img: np.ndarray, starts: np.ndarray, ends: np.ndarray, color: Color, thickness: int = 2, inplace: bool = False
) -> np.ndarray:
    """Draw lines.

    Note: modifies image in-place. Send in a copy if you don't want your image modified.
    """
    if not inplace:
        img = img.copy()

    for pt1, pt2 in zip(starts, ends):
        cv2.line(img, pt1.astype(int), pt2.astype(int), color=color, thickness=thickness)

    return img
