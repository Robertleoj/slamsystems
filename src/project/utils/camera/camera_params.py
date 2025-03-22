"""Module for camera model utilities

TODO: Move the pnp stuff elsewhere
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

import project.foundation.utils as cpp_utils
from project.utils.spatial.pose import Pose, to_homogeneous_2D


@dataclass(frozen=True, kw_only=True)
class Intrinsics:
    width: int
    height: int
    camera_matrix: np.ndarray
    distortion_parameters: np.ndarray

    def is_undistorted(self):
        return np.allclose(self.distortion_parameters, np.zeros_like(self.distortion_parameters))

    @staticmethod
    def from_pinhole_params(
        width: int, height: int, fx: float, fy: float, cx: float, cy: float, distortion_params: np.ndarray
    ) -> Intrinsics:
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        return Intrinsics(
            width=width, height=height, camera_matrix=camera_matrix, distortion_parameters=distortion_params
        )

    @property
    def fx(self):
        return self.camera_matrix[0, 0]

    @property
    def fy(self):
        return self.camera_matrix[1, 1]

    @property
    def cx(self):
        return self.camera_matrix[0, 2]

    @property
    def cy(self):
        return self.camera_matrix[1, 2]

    def fov(self) -> tuple[float, float]:
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        fov_x = 2 * np.arctan(self.width / (2 * fx))
        fov_y = 2 * np.arctan(self.height / (2 * fy))
        return fov_x, fov_y

    def aspect_ratio(self) -> float:
        return self.width / self.height

    def to_json_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_parameters": self.distortion_parameters.tolist(),
        }

    @staticmethod
    def from_json_dict(json_dict: dict) -> Intrinsics:
        return Intrinsics(
            width=json_dict["width"],
            height=json_dict["height"],
            camera_matrix=np.array(json_dict["camera_matrix"]),
            distortion_parameters=np.array(json_dict["distortion_parameters"]),
        )

    def normalize_pixels(self, points: np.ndarray) -> np.ndarray:
        assert len(points.shape) == 2 and points.shape[1] == 2

        return cv2.undistortPoints(
            points.astype(np.float64), self.camera_matrix, self.distortion_parameters, None
        ).squeeze()

    def unproject_depths(self, pixels: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """Unproject pixels to 3D coordinates given their depths

        Args:
            pixels: the pixels to unproject, shape (N x 2)
            depths: depths of the pixels, shape (N,)

        Returns:
            unprojected_points: The 3D location of the unprojected points, shape (N x 3)
        """
        assert len(depths.shape) == 1
        assert depths.shape[0] == pixels.shape[0], f"pixels: {pixels.shape}, depths: {depths.shape}"
        assert pixels.shape[1] == 2

        normalized_pixels = self.normalize_pixels(pixels)
        normalized_pixels_h = to_homogeneous_2D(normalized_pixels)
        return normalized_pixels_h * depths[:, None]

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.undistort(image, self.camera_matrix, self.distortion_parameters)

    def project_points(self, points_in_world: np.ndarray, world_to_cam: Pose = Pose.identity()) -> np.ndarray:
        """Project points into the camera

        Args:
            points_in_cam: N x 3 array of points in camera space

        Returns:
            pixels: N x 2 array of the pixel coordinates of the projected points
        """
        assert len(points_in_world.shape) == 2
        assert points_in_world.shape[1] == 3

        cam_to_world = world_to_cam.inv

        projected_pixels, _ = cv2.projectPoints(
            points_in_world,
            cam_to_world.rvec,
            cam_to_world.tvec,
            self.camera_matrix,
            self.distortion_parameters,
        )  # type: ignore

        return projected_pixels.reshape(-1, 2)

    def undistort_pixels(self, pixels: np.ndarray) -> np.ndarray:
        return cv2.undistortImagePoints(pixels, self.camera_matrix, self.distortion_parameters)

    def to_undistorted(self) -> Intrinsics:
        return replace(self, distortion_parameters=np.zeros_like(self.distortion_parameters))

    def to_cpp(self) -> cpp_utils.CameraParams:
        return cpp_utils.CameraParams(self.camera_matrix, self.distortion_parameters, self.width, self.height)

    def resized(self, width: int, height: int):
        scale_x = width / self.width
        scale_y = height / self.height

        K_new = self.camera_matrix.copy()

        K_new[0, 0] *= scale_x  # fx'
        K_new[1, 1] *= scale_y  # fy'
        K_new[0, 2] *= scale_x  # cx'
        K_new[1, 2] *= scale_y  # cy'

        return Intrinsics(
            width=width, height=height, camera_matrix=K_new, distortion_parameters=self.distortion_parameters
        )


@dataclass(frozen=True)
class Camera:
    intrinsics: Intrinsics
    extrinsics: Pose

    def project_points(self, points_in_world: np.ndarray) -> np.ndarray:
        return self.intrinsics.project_points(points_in_world, self.extrinsics)
