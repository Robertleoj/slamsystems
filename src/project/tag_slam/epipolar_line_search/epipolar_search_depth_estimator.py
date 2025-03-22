import numpy as np

from project.utils.camera.camera_params import Intrinsics
from project.utils.depth_estimation.block_search import epipolar_line_search_block_match
from project.utils.image import to_greyscale
from project.utils.spatial import Pose
from project.utils.spatial.pose import to_homogeneous_2D
from project.utils.triangulation import triangulate_normalized_image_points


class KeyFrameDepthEstimator:
    undistorted_image: np.ndarray
    undistorted_grey_image: np.ndarray
    world_to_cam: Pose
    pixels: np.ndarray
    normalized_pixels: np.ndarray
    depth_estimates: np.ndarray
    undistorted_intrinsics: Intrinsics

    def __init__(self, undistorted_intrinsics: Intrinsics, world_to_cam: Pose, undistorted_image: np.ndarray) -> None:
        assert undistorted_intrinsics.is_undistorted()
        self.undistorted_image = undistorted_image
        self.undistorted_grey_image = to_greyscale(undistorted_image)

        self.pixels = self._get_pixels(undistorted_intrinsics)
        self.normalized_pixels = undistorted_intrinsics.normalize_pixels(self.pixels)

        self.world_to_cam = world_to_cam
        self.undistorted_intrinsics = undistorted_intrinsics

        self.depth_estimates = np.zeros(self.pixels.shape[0], dtype=np.float32)
        self.depth_estimate_counts = np.zeros(self.pixels.shape[0], dtype=np.int32)

    def _get_pixels(self, undistorted_intrinsics: Intrinsics) -> np.ndarray:
        x_to_match = np.linspace(0, undistorted_intrinsics.width - 1, 200)
        y_to_match = np.linspace(0, undistorted_intrinsics.height - 1, 150)

        # Create a grid of coordinates
        X, Y = np.meshgrid(x_to_match, y_to_match)

        # Stack them into a list of coordinates
        return np.vstack([X.ravel(), Y.ravel()]).T.astype(np.int64)

    def add_view(self, world_to_cam: Pose, undistorted_frame: np.ndarray):
        undistorted_grey_frame = to_greyscale(undistorted_frame)

        key_to_cam = self.world_to_cam.inv @ world_to_cam

        matched_pixels, valid_mask = epipolar_line_search_block_match(
            self.pixels,
            self.undistorted_grey_image,
            undistorted_grey_frame,
            key_to_cam,
            self.undistorted_intrinsics,
            block_size=7,
            undistort=False,
            search_zone_size=300,
        )

        valid_indices = np.where(valid_mask)[0]
        pixels1_normalized = self.normalized_pixels[valid_indices]

        pixels2 = matched_pixels[valid_indices]
        pixels2_normalized = self.undistorted_intrinsics.normalize_pixels(pixels2)

        cam1_to_cam2 = self.world_to_cam.inv @ world_to_cam

        points3d_cam1 = triangulate_normalized_image_points(
            [pixels1_normalized, pixels2_normalized], [Pose.identity(), cam1_to_cam2]
        )

        depths = points3d_cam1[:, 2]

        valid_mask = (depths > 100) & (depths < 10 * 100 * 10)

        valid_indices = valid_indices[valid_mask]
        depths = depths[valid_mask]

        self.depth_estimate_counts[valid_indices] += 1
        n = self.depth_estimate_counts[valid_indices]

        self.depth_estimates[valid_indices] = ((n - 1) / n) * self.depth_estimates[valid_indices] + (depths / n)

    def get_colored_points_in_world(self) -> tuple[np.ndarray, np.ndarray] | None:
        valid_pixel_mask = self.depth_estimate_counts > 5

        if valid_pixel_mask.sum() == 0:
            return None

        pixels = self.pixels[valid_pixel_mask]

        pixel_colors = self.undistorted_image[pixels[:, 1], pixels[:, 0]]

        points3d_cam = (
            to_homogeneous_2D(self.normalized_pixels[valid_pixel_mask])
            * self.depth_estimates[valid_pixel_mask][:, None]
        )
        return self.world_to_cam.apply(points3d_cam), pixel_colors
