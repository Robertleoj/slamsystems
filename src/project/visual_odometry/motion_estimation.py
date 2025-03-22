import cv2
import numpy as np

from project.utils.camera.camera_params import Intrinsics


def estimate_motion_from_matches(pixels1: np.ndarray, pixels2: np.ndarray, intrinsics: Intrinsics):
    """
    see https://stackoverflow.com/questions/77522308/understanding-cv2-recoverposes-coordinate-frame-transformations
    for description of R and t. These are
    cam2_T_cam_1 = [ R   t ]
                   [ 0   1 ]

    """
    camera_matrix = intrinsics.camera_matrix
    dist_coeffs = intrinsics.distortion_parameters

    pixels1_undistorted = cv2.undistortImagePoints(pixels1, camera_matrix, dist_coeffs, None)
    pixels2_undistorted = cv2.undistortImagePoints(pixels2, camera_matrix, dist_coeffs, None)

    E, mask = cv2.findEssentialMat(
        pixels1_undistorted, pixels2_undistorted, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )

    mask = mask.reshape(-1)

    _, R, t, mask = cv2.recoverPose(E, pixels1_undistorted, pixels2_undistorted)

    return R, t
