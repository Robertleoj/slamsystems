from __future__ import annotations

from typing import TypedDict, cast

import cv2
import numpy as np

from project.utils.camera.camera_params import Intrinsics
from project.utils.colors import Colors
from project.utils.drawing import draw_lines, draw_pixels
from project.utils.spatial.pose import Pose


class _PnPInitialGuessArgs(TypedDict):
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    useExtrinsicGuess: bool


def _pnp_initial_guess_args_from_world_to_cam(world_to_cam: Pose | None) -> _PnPInitialGuessArgs:
    rvec = None
    tvec = None
    if world_to_cam is not None:
        cam_to_world = world_to_cam.inv
        rvec = cam_to_world.rvec
        tvec = cam_to_world.tvec

    return {"rvec": rvec, "tvec": tvec, "useExtrinsicGuess": world_to_cam is not None}


def _calculate_reprojection_errors(
    intrinsics: Intrinsics,
    world_points: np.ndarray,
    pixel_coordinates: np.ndarray,
    world_to_cam: Pose,
    debug_img: np.ndarray | None = None,
) -> np.ndarray:
    reprojected = intrinsics.project_points(world_points, world_to_cam)

    repr_error = np.linalg.norm(reprojected - pixel_coordinates, axis=1)

    if debug_img is not None:
        _draw_pnp_debug(intrinsics, debug_img, pixel_coordinates, pixels_reprojected=reprojected)

    return repr_error


def estimate_pose_pnp(
    intrinsics: Intrinsics,
    world_points: np.ndarray,
    pixel_coordinates: np.ndarray,
    world_to_cam_initial_guess: Pose | None = None,
    method: int = cv2.SOLVEPNP_ITERATIVE,
    debug_img: np.ndarray | None = None,
) -> tuple[Pose, np.ndarray]:
    """Estimate camera pose with PnP

    Args:
        intrinsics: Camera intrinsics
        world_points: The world points, shape N x 3,
        pixel_coordinates: Pixel coordinates corresponding to the world points, shape (N x 3)
        world_to_cam_initial_guess: initial guess for the camera pose
        method: How to solve it.
            Don't pass in raw integers, use the cv2.SOLVEPNP_<something> options.


    Returns:
        estimated_pose: Estimated pose of the camera.
        reprojection_errors: reprojection error for each point
    """

    success, rvec, tvec = cv2.solvePnP(
        world_points,
        pixel_coordinates,
        intrinsics.camera_matrix,
        intrinsics.distortion_parameters,
        **_pnp_initial_guess_args_from_world_to_cam(world_to_cam_initial_guess),
        flags=method,
    )

    if not success:
        raise RuntimeError("PnP failed")

    rot_mat, _ = cv2.Rodrigues(rvec)

    world_to_cam = Pose.from_rotmat_trans(rot_mat, tvec.reshape(3)).inv

    repr_errors = _calculate_reprojection_errors(
        intrinsics, world_points, pixel_coordinates, world_to_cam, debug_img=debug_img
    )

    return world_to_cam, repr_errors


def estimate_pose_pnp_ransac(
    intrinsics: Intrinsics,
    world_points: np.ndarray,
    pixel_coordinates: np.ndarray,
    world_to_cam_initial_guess: Pose | None = None,
    inlier_repojection_error_threshold: float = 8.0,
    num_iterations: int = 100,
    debug_img: np.ndarray | None = None,
) -> tuple[Pose, np.ndarray, np.ndarray]:
    """Estimate camera pose with Ransac PnP

    Args:
        world_points: The world points, shape N x 3,
        pixel_coordinates: Pixel coordinates corresponding to the world points, shape (N x 3)
        world_to_cam_initial_guess: Initial guess for the camera pose.

    Returns:
        estimated_pose: Estimated pose of the camera.
        reprojection_errors: reprojection error for each point, shape N.
        inlier_mask: shape N boolean mask denoting inliers
    """

    world_points = world_points.astype(float)
    pixel_coordinates = pixel_coordinates.astype(float)

    success, rvec, tvec, inlier_indices = cv2.solvePnPRansac(
        world_points,
        pixel_coordinates,
        intrinsics.camera_matrix,
        intrinsics.distortion_parameters,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=inlier_repojection_error_threshold,
        iterationsCount=num_iterations,
        **_pnp_initial_guess_args_from_world_to_cam(world_to_cam_initial_guess),
    )
    inlier_indices = cast(np.ndarray, inlier_indices)

    inlier_indices = inlier_indices.reshape(-1)
    inlier_mask = np.zeros(world_points.shape[0], dtype=bool)
    inlier_mask[inlier_indices] = True

    if not success:
        raise RuntimeError("PnP failed")

    rot_mat, _ = cv2.Rodrigues(rvec)

    world_to_cam = Pose.from_rotmat_trans(rot_mat, tvec.reshape(3)).inv

    repr_errors = _calculate_reprojection_errors(
        intrinsics, world_points, pixel_coordinates, world_to_cam, debug_img=debug_img
    )

    return world_to_cam, repr_errors, inlier_mask


def _draw_pnp_debug(self, img: np.ndarray, pixels_matched: np.ndarray, pixels_reprojected: np.ndarray):
    draw_pixels(img, pixels_matched, color=Colors.GREEN, radius=3, inplace=True)
    draw_pixels(img, pixels_reprojected, Colors.RED, radius=3, inplace=True)
    draw_lines(img, pixels_matched, pixels_reprojected, color=Colors.WHITE, thickness=1, inplace=True)
