import logging
from collections import defaultdict

import numpy as np

import project.foundation.symforce_exercises.depth_slam as cpp_ds
from project.depth_slam.data_types import Frame, Map, get_features_3d
from project.utils.camera.camera_params import Intrinsics
from project.utils.pnp import estimate_pose_pnp
from project.utils.spatial.pose import Pose

LOG = logging.getLogger(__name__)


def delete_bad_landmarks(map: Map, frames: list[Frame], keyframe_indices: list[int]):
    obs_counts = defaultdict(int)
    for keyframe_idx in keyframe_indices:
        for landmark_id in frames[keyframe_idx].features.keys():
            obs_counts[landmark_id] += 1

    to_delete = []

    for landmark_id in map.landmarks.keys():
        if obs_counts[landmark_id] <= 1 and landmark_id not in frames[-1].features:
            to_delete.append(landmark_id)

    for landmark_id in to_delete:
        for frame in frames:
            if landmark_id in frame.features:
                del frame.features[landmark_id]
        del map.landmarks[landmark_id]

    LOG.info(f"deleted {len(to_delete)} landmarks!")

    return obs_counts


def optimize(camera_calibration: Intrinsics, map: Map, frames: list[Frame], keyframe_indices: list[int]):
    landmark_obs_counts = delete_bad_landmarks(map, frames, keyframe_indices)

    camera_poses: list[cpp_ds.CameraPose] = []
    observations: list[cpp_ds.LandmarkProjectionObservation] = []
    landmarks: list[cpp_ds.Landmark] = []

    seen_landmarks = set()

    non_optimized_landmarks = set()

    for keyframe_idx in keyframe_indices:
        frame = frames[keyframe_idx]
        cam_pose_cpp = cpp_ds.CameraPose(keyframe_idx, frame.camera_pose.mat.astype(np.float64))
        camera_poses.append(cam_pose_cpp)

        for feature in frame.features.values():
            if landmark_obs_counts[feature.landmark_id] <= 1:
                non_optimized_landmarks.add(feature.landmark_id)
                continue

            undistorted_pixel = camera_calibration.undistort_pixels(feature.pixel.reshape(-1, 2)).reshape(2)

            observation = cpp_ds.LandmarkProjectionObservation(
                keyframe_idx, feature.landmark_id, undistorted_pixel.astype(np.float64), feature.depth
            )
            observations.append(observation)
            seen_landmarks.add(feature.landmark_id)

    for lid in seen_landmarks:
        landmark = cpp_ds.Landmark(lid, map.landmarks[lid].loc)
        landmarks.append(landmark)

    optimized_camera_poses, optimized_landmarks = cpp_ds.depth_slam_ba(
        camera_calibration.to_undistorted().to_cpp(), camera_poses, observations, landmarks
    )

    for cam_pose_cpp in optimized_camera_poses:
        frame = frames[cam_pose_cpp.frame_id]

        old_cam_pose = frame.camera_pose
        new_cam_pose = Pose(cam_pose_cpp.camera_pose.copy())

        old_to_new = new_cam_pose @ old_cam_pose.inv

        # if a landmark with observation count 1 was observed in
        # this frame, we want to transform it
        for lid in non_optimized_landmarks:
            if lid in frame.features:
                old_pos_in_world = map.landmarks[lid].loc.reshape(1, 3)
                new_pos_in_world = old_to_new.apply(old_pos_in_world)
                map.landmarks[lid].loc = new_pos_in_world.reshape((3,))

        frame.camera_pose = new_cam_pose

    for landmark in optimized_landmarks:
        map.landmarks[landmark.id].loc = landmark.loc.copy()

    # relocalize all frames
    for frame_id, frame in enumerate(frames):
        if frame_id in keyframe_indices:
            continue

        feature_pixels = frame.feature_pixels
        features3d = get_features_3d(map, frame)

        new_frame_pose, _ = estimate_pose_pnp(camera_calibration, features3d, feature_pixels, frame.camera_pose)

        frame.camera_pose = new_frame_pose

    LOG.info("optimized camera poses and landmarks")
