from dataclasses import dataclass, replace

import dt_apriltags as april
import numpy as np

from project.foundation.tag_slam import CameraPose, Tag, TagObservation, tag_slam_ba
from project.tag_slam.epipolar_line_search.epipolar_search_depth_estimator import KeyFrameDepthEstimator
from project.utils.camera.camera_params import Intrinsics
from project.utils.depth_estimation.block_search import epipolar_line_search, epipolar_line_search_block_match
from project.utils.image import to_greyscale
from project.utils.markers import DetectedTag, Tag3D, detect_tags, estimate_3d_tag, get_tag_detector
from project.utils.pnp import estimate_pose_pnp
from project.utils.spatial import Pose
from project.utils.spatial.pose import to_homogeneous_2D
from project.utils.triangulation import triangulate_normalized_image_points


class TagSlam:
    # mapping from tag id to the current believed
    # pose of the tag
    landmarks: dict[int, Tag3D]
    detections: list[dict[int, DetectedTag]]

    keyframe_frequency: int
    undistorted_keyframe_imgs: dict[int, np.ndarray]
    keyframe_indices: list[int]
    keyframe_depth_estimators: dict[int, KeyFrameDepthEstimator] | None

    # camera pose for each frame
    camera_poses: list[Pose]

    # the side lengths of the tags
    tag_side_length: float

    tag_detector: april.Detector

    cam_intrinsics: Intrinsics

    def __init__(
        self,
        tag_side_length: float,
        cam_intrinsics: Intrinsics,
        keyframe_frequency: int = 25,
        estimate_dense: bool = False,
    ) -> None:
        self.tag_side_length = tag_side_length
        self.tag_detector = get_tag_detector()
        self.detections = []
        self.landmarks = {}
        self.camera_poses = []
        self.cam_intrinsics = cam_intrinsics
        self.undistorted_intrinsics = cam_intrinsics.to_undistorted()
        self.keyframe_frequency = keyframe_frequency
        self.undistorted_keyframe_imgs = {}
        self.keyframe_indices = []

        self.keyframe_depth_estimators = None
        if estimate_dense:
            self.keyframe_depth_estimators = {}

    def process_frame(self, frame: np.ndarray) -> None:
        undistorted_frame = self.cam_intrinsics.undistort_image(frame)

        frame_idx = len(self.camera_poses)
        is_keyframe = False
        if frame_idx % self.keyframe_frequency == 0:
            self.keyframe_indices.append(frame_idx)
            self.undistorted_keyframe_imgs[frame_idx] = undistorted_frame
            is_keyframe = True

        detected_tags = detect_tags(undistorted_frame, self.tag_detector)
        self.detections.append(detected_tags)

        if len(self.camera_poses) == 0:
            estimated_3d_tags = {
                tag_id: estimate_3d_tag(tag, self.undistorted_intrinsics, self.tag_side_length)
                for tag_id, tag in detected_tags.items()
            }

            # first frame - initialize map and cam pose
            self.landmarks.update(estimated_3d_tags)
            self.camera_poses.append(Pose.identity())

            if self.keyframe_depth_estimators is not None:
                self.keyframe_depth_estimators[frame_idx] = KeyFrameDepthEstimator(
                    self.undistorted_intrinsics, Pose.identity(), undistorted_frame
                )

            return

        predicted_pose = self.predict_camera_pose()

        # find camera pose using tags
        estimated_camera_pose = self.estimate_camera_pose(detected_tags, predicted_pose)

        # add new tags to map
        for tag_id, tag in detected_tags.items():
            if tag_id in self.landmarks:
                continue

            tag3d = estimate_3d_tag(tag, self.undistorted_intrinsics, self.tag_side_length).transform(
                estimated_camera_pose
            )

            self.landmarks[tag_id] = tag3d

        self.camera_poses.append(estimated_camera_pose)

        if self.keyframe_depth_estimators is not None:
            if is_keyframe:
                self.keyframe_depth_estimators[frame_idx] = KeyFrameDepthEstimator(
                    self.undistorted_intrinsics, estimated_camera_pose, undistorted_frame
                )
            else:
                curr_keyframe_idx = self.keyframe_indices[-1]
                curr_keyframe_pose = self.camera_poses[curr_keyframe_idx]
                if np.linalg.norm(estimated_camera_pose.tvec - curr_keyframe_pose.tvec) > 20:
                    self.keyframe_depth_estimators[curr_keyframe_idx].add_view(
                        estimated_camera_pose,
                        undistorted_frame,
                    )

    def predict_camera_pose(self) -> Pose:
        return self.camera_poses[-1]

    def estimate_camera_pose(self, detected_tags: dict[int, DetectedTag], predicted: Pose) -> Pose:
        common_ids = list(self.landmarks.keys() & detected_tags.keys())

        # one 3D point for each tag corner
        num_points = len(common_ids) * 4
        world_points = np.zeros((num_points, 3))
        cam_points = np.zeros((num_points, 2))

        for i, tag_id in enumerate(common_ids):
            world_points[4 * i : 4 * (i + 1)] = self.landmarks[tag_id].corners
            cam_points[4 * i : 4 * (i + 1)] = detected_tags[tag_id].corners

        cam_in_world, _ = estimate_pose_pnp(self.undistorted_intrinsics, world_points, cam_points, predicted)

        return cam_in_world

    def refine(self) -> None:
        N = len(self.camera_poses)

        camera_poses_in = []
        tags_in = []
        observations_in = []

        camera_cal = np.array(
            [
                self.undistorted_intrinsics.fx,
                self.undistorted_intrinsics.fy,
                self.undistorted_intrinsics.cx,
                self.undistorted_intrinsics.cy,
            ]
        )

        detected_tags: set[int] = set()

        for frame_idx in self.keyframe_indices:
            camera_poses_in.append(CameraPose(frame_idx, self.camera_poses[frame_idx].mat))

            for tag_id, tag_det in self.detections[frame_idx].items():
                detected_tags.add(tag_id)
                observations_in.append(TagObservation(frame_idx, tag_id, tag_det.corners.flatten()))

        for tag_id in detected_tags:
            tag_pose = self.landmarks[tag_id].pose
            tags_in.append(Tag(tag_id, tag_pose.mat))

        camera_poses_out, tags_out = tag_slam_ba(
            camera_cal, camera_poses_in, observations_in, tags_in, self.tag_side_length
        )

        for cam_pose in camera_poses_out:
            self.camera_poses[cam_pose.frame_id] = Pose(cam_pose.pose.copy())

        for tag in tags_out:
            self.landmarks[tag.tag_id] = replace(self.landmarks[tag.tag_id], pose=Pose(tag.pose.copy()))

        for frame_idx in range(N):
            if frame_idx in self.keyframe_indices:
                continue

            new_cam_pose = self.estimate_camera_pose(self.detections[frame_idx], self.camera_poses[frame_idx])
            self.camera_poses[frame_idx] = new_cam_pose
