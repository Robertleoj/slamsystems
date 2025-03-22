from __future__ import annotations

import numpy as np
import rerun as rr

from project.depth_slam.backend import optimize
from project.depth_slam.data_types import Feature, Frame, Map, PointCloud, get_features_3d
from project.utils.camera.camera_params import Intrinsics
from project.utils.camera.view import View
from project.utils.colors import Colors
from project.utils.features import get_good_features_to_track
from project.utils.image import to_greyscale
from project.utils.optical_flow import lk_optical_flow
from project.utils.pnp import estimate_pose_pnp_ransac
from project.utils.spatial.pose import Pose


class DepthSlam:
    def __init__(self, cam_intrinsics: Intrinsics):
        self.intrinsics = cam_intrinsics
        self.frames: list[Frame] = []
        self.keyframe_indices: list[int] = []
        self.keyframe_views: dict[int, View] = {}
        self.keyframe_pointclouds_in_cam: dict[int, PointCloud] = {}
        self.map = Map()
        self.curr_frame_idx = 0
        self.last_frame: np.ndarray | None = None

    def predict_curr_pose(self) -> Pose:
        if len(self.frames) == 0:
            return Pose.identity()

        return self.frames[-1].camera_pose

    def get_feature_mask(self, frame: Frame):
        radius = 5

        shape = (self.intrinsics.height, self.intrinsics.width)

        mask = np.ones((self.intrinsics.height, self.intrinsics.width))

        mask = np.ones(shape, dtype=bool)  # Start with True everywhere
        yy, xx = np.indices(shape)

        for x, y in frame.feature_pixels:
            dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
            mask[dist <= radius] = False  # Set False within radius

        return mask

    def make_pointcloud(self, color: np.ndarray, depth: np.ndarray) -> PointCloud:
        y, x = np.where((depth > 0) & (depth < 500))

        valid_depths = depth[y, x]
        valid_colors = color[y, x]
        pixels = np.array([x, y]).T

        points3d_cam = self.intrinsics.unproject_depths(pixels, valid_depths)

        num_to_keep = 10000
        step = points3d_cam.shape[0] // num_to_keep

        return PointCloud(points3d_cam[::step], valid_colors[::step])

    def make_keyframe(self, color: np.ndarray, depth: np.ndarray, frame: Frame | None = None) -> Frame:
        """Create a keyframe."""

        rr.log("/keyframe_gen/", rr.Image(color))

        all_features = {}
        cam_pose = Pose.identity()
        feature_mask = None
        if frame is not None:
            all_features = frame.features
            cam_pose = frame.camera_pose
            feature_mask = self.get_feature_mask(frame)

        # create features to track
        grey = to_greyscale(color)

        new_feature_pixels = get_good_features_to_track(grey, 200, quality_level=0.1, mask=feature_mask)

        rr.log("/keyframe_gen/features", rr.Points2D(new_feature_pixels, colors=Colors.RED, radii=10.0))

        # triangulate the features
        feature_point_depths = depth[new_feature_pixels[:, 1].astype(int), new_feature_pixels[:, 0].astype(int)]

        valid_depth_mask = feature_point_depths > 0

        valid_depths = feature_point_depths[valid_depth_mask]
        valid_features = new_feature_pixels[valid_depth_mask]

        rr.log("/keyframe_gen/valid_features", rr.Points2D(valid_features, colors=Colors.GREEN, radii=10))

        features3d_in_cam = self.intrinsics.unproject_depths(valid_features, valid_depths)

        features3d_in_world = cam_pose.apply(features3d_in_cam)

        # add them to the map
        landmark_ids = []
        for point in features3d_in_world:
            landmark_ids.append(self.map.add_landmark(point))

        for landmark_id, valid_feature, valid_depth in zip(landmark_ids, valid_features, valid_depths):
            all_features[landmark_id] = Feature(landmark_id, valid_feature, depth=float(valid_depth))

        self.keyframe_indices.append(self.curr_frame_idx)
        self.keyframe_views[self.curr_frame_idx] = View(color=color, depth=depth, intrinsics=self.intrinsics)

        pc = self.make_pointcloud(color, depth)
        self.keyframe_pointclouds_in_cam[self.curr_frame_idx] = pc

        return Frame(all_features, cam_pose)

    def make_curr_frame(self, prev_img: np.ndarray, prev_frame: Frame, curr_img: np.ndarray, curr_depth: np.ndarray):
        predicted_pose = self.predict_curr_camera_pose()

        prev_feature_pixels = prev_frame.feature_pixels
        prev_features3d = get_features_3d(self.map, prev_frame)

        (curr_feature_pixels, success_mask, _err) = lk_optical_flow(prev_img, curr_img, prev_feature_pixels)

        success_indices = np.where(success_mask)[0]

        successfully_tracked_feature_pixels = curr_feature_pixels[success_indices]

        successfully_tracked_prev_features3d = prev_features3d[success_mask]

        # now localize the curr camera
        estimated_pose, _repr_error, inlier_mask = estimate_pose_pnp_ransac(
            self.intrinsics,
            successfully_tracked_prev_features3d,
            successfully_tracked_feature_pixels,
            predicted_pose,
        )

        success_indices = success_indices[inlier_mask]
        curr_frame_features: dict[int, Feature] = {}
        for idx in success_indices:
            feature_pixel = curr_feature_pixels[idx]
            landmark_id = prev_frame.landmark_ids[idx]

            curr_frame_features[landmark_id] = Feature(landmark_id, feature_pixel)

        curr_frame = Frame(curr_frame_features, estimated_pose)

        if len(success_indices) < 40:
            curr_frame = self.make_keyframe(curr_img, curr_depth, curr_frame)

        rr.log("/frames", rr.Image(curr_img))
        rr.log(
            "/frames/features",
            rr.Points2D(np.array([f.pixel for f in curr_frame_features.values()]), radii=10.0, colors=Colors.GREEN),
        )

        return curr_frame

    def predict_curr_camera_pose(self) -> Pose:
        if self.curr_frame_idx == 0:
            return Pose.identity()

        return self.frames[self.curr_frame_idx - 1].camera_pose

    def process_frame(self, color: np.ndarray, depth: np.ndarray):
        if len(self.frames) == 0:
            frame = self.make_keyframe(color, depth)
            self.frames.append(frame)
            self.last_frame = color
            self.curr_frame_idx += 1
            return

        assert self.last_frame is not None

        # Now we can assume this is not the first frame
        last_frame = self.frames[self.curr_frame_idx - 1]
        curr_frame = self.make_curr_frame(self.last_frame, last_frame, color, depth)

        self.frames.append(curr_frame)
        self.last_frame = color

        if self.curr_frame_idx == self.keyframe_indices[-1] and self.curr_frame_idx > 0:
            optimize(self.intrinsics, self.map, self.frames, self.keyframe_indices)

        self.curr_frame_idx += 1
