from dataclasses import replace
from itertools import pairwise

import cv2
import dt_apriltags as april
import numpy as np
from tqdm import tqdm

from project.foundation.tag_slam import CameraPose, Tag, TagObservation, tag_slam_ba
from project.utils.camera.camera_params import Intrinsics
from project.utils.image import to_greyscale
from project.utils.markers import DetectedTag, Tag3D, detect_tags, estimate_3d_tag, get_tag_detector
from project.utils.pnp import estimate_pose_pnp
from project.utils.spatial import Pose
from project.utils.triangulation import check_epipolar_constraint, triangulate_normalized_image_points


class TagSlam:
    # mapping from tag id to the current believed
    # pose of the tag
    landmarks: dict[int, Tag3D]
    detections: list[dict[int, DetectedTag]]

    keyframe_frequency: int
    keyframe_imgs: dict[int, np.ndarray]
    keyframe_indices: list[int]

    # camera pose for each frame
    camera_poses: list[Pose]

    # the side lengths of the tags
    tag_side_length: float

    tag_detector: april.Detector

    cam_intrinsics: Intrinsics

    def __init__(self, tag_side_length: float, cam_intrinsics: Intrinsics, keyframe_frequency: int = 25) -> None:
        self.tag_side_length = tag_side_length
        self.tag_detector = get_tag_detector()
        self.detections = []
        self.landmarks = {}
        self.camera_poses = []
        self.cam_intrinsics = cam_intrinsics
        self.keyframe_frequency = keyframe_frequency
        self.keyframe_imgs = {}
        self.keyframe_indices = []

    def process_frame(self, frame: np.ndarray) -> None:
        frame_idx = len(self.camera_poses)
        if frame_idx % self.keyframe_frequency == 0:
            self.keyframe_indices.append(frame_idx)
            self.keyframe_imgs[frame_idx] = frame

        detected_tags = detect_tags(frame, self.tag_detector)
        self.detections.append(detected_tags)

        if len(self.camera_poses) == 0:
            estimated_3d_tags = {
                tag_id: estimate_3d_tag(tag, self.cam_intrinsics, self.tag_side_length)
                for tag_id, tag in detected_tags.items()
            }

            # first frame - initialize map and cam pose
            self.landmarks.update(estimated_3d_tags)
            self.camera_poses.append(Pose.identity())
            return

        predicted_pose = self.predict_camera_pose()

        # find camera pose using tags
        estimated_camera_pose = self.estimate_camera_pose(detected_tags, predicted_pose)

        # add new tags to map
        for tag_id, tag in detected_tags.items():
            if tag_id in self.landmarks:
                continue

            tag3d = estimate_3d_tag(tag, self.cam_intrinsics, self.tag_side_length).transform(estimated_camera_pose)

            self.landmarks[tag_id] = tag3d

        self.camera_poses.append(estimated_camera_pose)

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

        cam_in_world, _ = estimate_pose_pnp(self.cam_intrinsics, world_points, cam_points, predicted)

        return cam_in_world

    def refine(self) -> None:
        N = len(self.camera_poses)

        camera_poses_in = []
        tags_in = []
        observations_in = []

        camera_cal = np.array(
            [self.cam_intrinsics.fx, self.cam_intrinsics.fy, self.cam_intrinsics.cx, self.cam_intrinsics.cy]
        )

        detected_tags: set[int] = set()

        for frame_idx in self.keyframe_indices:
            camera_poses_in.append(CameraPose(frame_idx, self.camera_poses[frame_idx].mat))

            for tag_id, tag_det in self.detections[frame_idx].items():
                detected_tags.add(tag_id)
                undistorted_corners = self.cam_intrinsics.undistort_pixels(tag_det.corners)

                observations_in.append(TagObservation(frame_idx, tag_id, undistorted_corners.flatten()))

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

    def get_dense_reconstruction_vo(self):
        keyframe_idx_to_colored_points = {}
        undistorted_intrinsics = self.cam_intrinsics.to_undistorted()
        deepflow = cv2.optflow.createOptFlow_DeepFlow()

        for keyframe_idx1, keyframe_idx2 in tqdm(pairwise(self.keyframe_indices)):
            frame1_distorted = self.keyframe_imgs[keyframe_idx1]
            frame2_distorted = self.keyframe_imgs[keyframe_idx2]

            cam1_pose = self.camera_poses[keyframe_idx1]
            cam2_pose = self.camera_poses[keyframe_idx2]

            frame1 = self.cam_intrinsics.undistort_image(frame1_distorted)
            frame2 = self.cam_intrinsics.undistort_image(frame2_distorted)

            grey1 = to_greyscale(frame1)
            grey2 = to_greyscale(frame2)

            flow = deepflow.calc(grey1, grey2, None)  # type: ignore

            y, x = np.where(np.linalg.norm(flow, axis=-1) > 1.0)

            pixels1 = np.array([x, y]).T
            pixels2 = pixels1 + flow[y, x]

            pixels1_normalized = undistorted_intrinsics.normalize_pixels(pixels1)
            pixels2_normalized = undistorted_intrinsics.normalize_pixels(pixels2)

            valid = check_epipolar_constraint(pixels1_normalized, pixels2_normalized, cam1_pose.inv @ cam2_pose, 1e-1)
            print(f"{valid.sum()}/{valid.shape[0]} valid points under epipolar constraint")

            pixels1_normalized = pixels1_normalized[valid]
            pixels2_normalized = pixels2_normalized[valid]

            points3d = triangulate_normalized_image_points(
                [pixels1_normalized, pixels2_normalized], [cam1_pose, cam2_pose]
            )

            points3d_in_cam1 = cam1_pose.inv.apply(points3d)
            points3d = points3d[points3d_in_cam1[:, 2] > 0]

            if points3d.shape[0] == 0:
                continue

            proj = undistorted_intrinsics.project_points(points3d, cam1_pose)

            x, y = proj.T.astype(int)
            valid = (x > 0) & (x < frame1.shape[1]) & (y > 0) & (y < frame1.shape[0])
            x = x[valid]
            y = y[valid]
            proj = proj[valid]

            points3d = points3d[valid]
            pixel_values = frame1[y, x]

            keyframe_idx_to_colored_points[keyframe_idx1] = (points3d, pixel_values)

        return keyframe_idx_to_colored_points
