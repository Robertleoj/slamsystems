import numpy as np

from project.utils.spatial.pose import Pose, skew


def essential_matrix(cam1_to_cam2: Pose) -> np.ndarray:
    cam2_to_cam1 = cam1_to_cam2.inv
    tvec = cam2_to_cam1.tvec
    rmat = cam2_to_cam1.rot_mat

    return skew(tvec) @ rmat


def fundamental_matrix(cam1_to_cam2: Pose, camera_matrix: np.ndarray) -> np.ndarray:
    E = essential_matrix(cam1_to_cam2)
    K = camera_matrix

    return np.linalg.inv(K.T) @ E @ np.linalg.inv(K)
