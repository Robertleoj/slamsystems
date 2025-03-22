from typing import Any

import einops
import numpy as np

from project.utils.camera.camera_params import Camera
from project.utils.camera.mvg import essential_matrix
from project.utils.spatial import Pose
from project.utils.spatial.pose import to_homogeneous_2D


def triangulate_points(
    rays: np.ndarray,
) -> np.ndarray:
    """Triangulate ray intersections.

    For each ray set, calculate the point that minimizes the sum of squared distances
    to the rays.

    Args:
        rays: Ray sets (N x K x 2 x 3), where N is the number of points to triangulate,
            K is the number of rays to use for each point, 2 means that two points
            define a ray, and 3 is 3D

    Returns:
        triangulated: shape (N x 3)
    """

    N = rays.shape[0]

    # Extract points from rays
    p1 = rays[:, :, 0, :]  # (N, K, 3)
    p2 = rays[:, :, 1, :]  # (N, K, 3)

    # Get ray directions
    d = p2 - p1  # (N, K, 3)
    d = d / np.linalg.norm(d, axis=2, keepdims=True)  # Normalize to unit vectors

    # Compute A and b for least squares
    A = np.eye(3)[None, None, :, :] - d[:, :, :, None] * d[:, :, None, :]
    b = (A @ p1[:, :, :, None]).squeeze(-1)  # (N, K, 3)

    # Solve least squares
    AtA = np.sum(A, axis=1)  # (N, 3, 3)
    assert AtA.shape == (N, 3, 3)

    Atb = np.sum(b, axis=1)  # (N, 3)
    assert Atb.shape == (N, 3)

    # Get optimal points
    triangulated = np.linalg.solve(AtA, Atb.reshape(N, 3, 1)).reshape(N, 3)  # (N, 3)

    return triangulated


def triangulate_normalized_image_points(
    camera_points: list[np.ndarray], camera_poses: list[Pose], debug_dict: dict[str, Any] | None = None
) -> np.ndarray:
    """Triangulate 3D locations of N matched image points from K cameras.

    Args:
        camera_points: Length K list of (N x 2/3) arrays corresponding to the matched
            points in each camera in normalized image coordinates (homogeneous or not).
        camera_poses: Length K list of the pose of each camera

    Returns:
        triangulated_points: shape (N x 3) array containing the triangulated location of each
            point.
    """
    assert len(camera_points) == len(camera_poses)
    assert len(camera_points) >= 2
    assert all(points.shape == camera_points[0].shape for points in camera_points)

    N = camera_points[0].shape[0]
    K = len(camera_points)

    camera_points = [to_homogeneous_2D(points) for points in camera_points]

    rays = np.zeros((N, K, 2, 3))

    for i, (pose, points) in enumerate(zip(camera_poses, camera_points)):
        # set ray starts, equal to camera origins
        rays[:, i, 0, :] = pose.tvec

        ray_directions_in_world = einops.einsum(pose.rot_mat, points, "h d, n d -> n h")

        ray_endpoints = ray_directions_in_world + pose.tvec

        rays[:, i, 1, :] = ray_endpoints

    triangulated_points = triangulate_points(rays)

    if debug_dict is not None:
        debug_dict.update(dict(rays=rays))

    return triangulated_points


def triangulate_pixels(
    pixels: list[np.ndarray], cameras: list[Camera], debug_dict: dict[str, Any] | None = None
) -> np.ndarray:
    """Triangulate 3D locations of N matched image points from K cameras.

    Args:
        camera_points: Length K list of (N x 2/3) arrays corresponding to the matched
            points in each camera in pixel coordinates.
        camera_poses: Length K list of cameras.

    Returns:
        triangulated_points: shape (N x 3) array containing the triangulated location of each
            point.
    """
    assert len(pixels) == len(cameras)
    assert len(pixels) >= 2
    assert all(pix.shape == pixels[0].shape for pix in pixels)

    normalized_pixels = []
    camera_poses = []
    for cam, pix in zip(cameras, pixels):
        normalized_pixels.append(cam.intrinsics.normalize_pixels(pix))
        camera_poses.append(cam.extrinsics)

    return triangulate_normalized_image_points(normalized_pixels, camera_poses, debug_dict)


def project_onto_epipolar_line(points1: np.ndarray, points2: np.ndarray, cam1_to_cam2):
    """
    Projects points in the second camera onto the corresponding epipolar line.

    Args:
        points1: (N, 3) array of normalized homogeneous points in the first camera.
        points2: (N, 3) array of normalized homogeneous points in the second camera.
        cam1_to_cam2: Camera transformation containing rotation and translation.

    Returns:
        projected_points2: (N, 3) array of points in the second camera projected onto epipolar lines.
    """

    points1 = to_homogeneous_2D(points1)
    points2 = to_homogeneous_2D(points2)

    R = cam1_to_cam2.rot_mat
    t = cam1_to_cam2.tvec

    # Compute essential matrix E = [t]_x R
    t_x = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = t_x @ R

    # Compute epipolar lines in the second camera
    lines2 = (E @ points1.T).T  # (N, 3)

    # Normalize the line equation (ax + by + c = 0)
    a, b, c = lines2[:, 0], lines2[:, 1], lines2[:, 2]
    norm_factor = np.sqrt(a**2 + b**2)
    a /= norm_factor
    b /= norm_factor
    c /= norm_factor

    # Project points2 onto their corresponding epipolar lines
    x2, y2 = points2[:, 0], points2[:, 1]
    lambda_factor = -(a * x2 + b * y2 + c) / (a**2 + b**2)  # Projection factor
    x_proj = x2 + lambda_factor * a
    y_proj = y2 + lambda_factor * b

    projected_points2 = np.stack([x_proj, y_proj], axis=1)

    return projected_points2


def check_epipolar_constraint(points1: np.ndarray, points2: np.ndarray, cam1_to_cam2: Pose, tol=1e-6):
    """
    Checks which points in the second camera are on the epipolar line defined by points in the first camera.

    Args:
        points1: (N, 3) array of normalized homogeneous points in the first camera.
        points2: (N, 3) array of normalized homogeneous points in the second camera.
        R: (3,3) rotation matrix from camera 1 to camera 2.
        t: (3,1) translation vector from camera 1 to camera 2.
        tol: Tolerance for checking the epipolar constraint.

    Returns:
        mask: (N,) boolean array where True means the point is on the epipolar line.
    """

    points1 = to_homogeneous_2D(points1)
    points2 = to_homogeneous_2D(points2)

    E = essential_matrix(cam1_to_cam2)

    # Compute epipolar lines in the second camera
    lines2 = (E @ points1.T).T  # (N, 3)

    # Check epipolar constraint x2^T * l2 = 0
    error = np.abs(np.sum(points2 * lines2, axis=1))  # (N,)

    return error < tol
