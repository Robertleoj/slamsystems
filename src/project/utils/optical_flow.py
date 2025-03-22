from typing import cast

import cv2
import numpy as np

from project.utils.image import to_greyscale


def lk_optical_flow(
    img1: np.ndarray, img2: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate LK flow

    Args:
        img1: First image
        img2: Second Image
        points: Points in image 1 to find in image 2

    Returns:
        points2: The tracked points in image 2
        success_mask: Mask specifying which points were successfully tracked
        err: The error for each point
    """
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    points = points.reshape(-1, 2)

    if len(img1.shape) == 3:
        img1 = to_greyscale(img1)
    if len(img2.shape) == 3:
        img2 = to_greyscale(img2)

    points2, status, err = cv2.calcOpticalFlowPyrLK(
        img1,
        img2,
        points.reshape(-1, 1, 2),
        None,  # type: ignore
    )  # type: ignore

    points2 = cast(np.ndarray, points2.squeeze())
    success_mask = cast(np.ndarray, status.squeeze() == 1)
    err = cast(np.ndarray, err.squeeze())

    return points2.squeeze(), success_mask, err
