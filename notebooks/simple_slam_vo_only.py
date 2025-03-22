# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Super simple SLAM using ICP VO only
# First, I'll use a video to use for experimentation
#
# The first version will:
# * At every frame, estimate the motion from the previous frame. Do this by
#     * Matching features
#     * Deprojecting the points
#     * Solving the transform between them, which is then the camera pose diff
# * Add the pointcloud from every frame to a map
# * Be able to display this map along with the camera movement in rerun.

# %% [markdown] vscode={"languageId": "plaintext"}
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2
import symforce
symforce.set_epsilon_to_symbol()
from project.camera_readers.realsense import RealSenseCamera
from project.utils.camera import View, Camera, Intrinsics, Video
from project.utils.spatial import Pose
import rerun as rr
import rerun.blueprint as rrb
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from project.utils.features import get_flann_matcher_for_orb, orb_feature_detect_and_match, draw_matches
from project.utils.pointclouds.align import align_point_matches_svd, align_point_matches_pypose, align_point_matches_svd_ransac
import tifffile as tiff
from itertools import pairwise
from pathlib import Path
import mediapy
import project.utils.rerun_utils as rr_utils
from project.utils.paths import repo_root
import json
import numpy as np

# %% [markdown]
# ## Capture video

# %%
vid = Video.load(repo_root() / 'data/videos/vid2')

# %%
# mediapy.show_videos([vid.color, vid.depth_assumed])

# %%
orb = cv2.ORB.create()
flann = get_flann_matcher_for_orb()


# %%
rr_utils.init("vo_icp")

# %%


def estimate_camera_movement_icp(color1: np.ndarray, depth1: np.ndarray, color2: np.ndarray, depth2: np.ndarray, intrinsics: Intrinsics, orb: cv2.ORB, flann: cv2.FlannBasedMatcher):
    
    match_result = orb_feature_detect_and_match(color1, color2, orb, flann, 20)
    matched_pixels1_depths = depth1[
        match_result.matched_points_1[:, 1].astype(int),
        match_result.matched_points_1[:, 0].astype(int)
    ]


    matched_pixels2_depths = depth2[
        match_result.matched_points_2[:, 1].astype(int),
        match_result.matched_points_2[:, 0].astype(int)
    ]

    valid_depth_mask = (matched_pixels1_depths > 0) & (matched_pixels2_depths > 0)

    matched_pixels1_depths = matched_pixels1_depths[valid_depth_mask]
    matched_pixels1 = match_result.matched_points_1[valid_depth_mask]

    matched_pixels2_depths = matched_pixels2_depths[valid_depth_mask]
    matched_pixels2 = match_result.matched_points_2[valid_depth_mask]

    points1 = intrinsics.unproject_depths(matched_pixels1, matched_pixels1_depths)
    points2 = intrinsics.unproject_depths(matched_pixels2, matched_pixels2_depths)


    cam2_pose, inlier_mask = align_point_matches_svd_ransac(points1, points2, 10)

    num_inliers = inlier_mask.sum()

    return cam2_pose, num_inliers

# %%
movements: list[Pose] = []
num_inliers_per_pose = []

for view1, view2 in tqdm(pairwise(vid)):
    movement, num_inliers = estimate_camera_movement_icp(
        view1.color, view1.depth_assumed, view2.color, view2.depth_assumed, vid.intrinsics, orb, flann
    )
    movements.append(movement)
    num_inliers_per_pose.append(num_inliers)

# %%
plt.hist(num_inliers_per_pose)

# %%
curr_pose = Pose.identity()
camera_poses = [curr_pose]
for movement in movements:
    curr_pose = curr_pose @ movement
    camera_poses.append(curr_pose)

# %%
len(camera_poses)

# %%
rr.log("/", rr.Clear(recursive=True))
rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
rr.reset_time()

# %%

for i, (pose, view) in enumerate(zip(camera_poses, vid)):
    t = i * 0.1

    if i % 5 == 0:
        depths = view.depth_assumed
        valid_depth_mask = depths > 0

        y, x = np.nonzero(valid_depth_mask)

        points3d = vid.intrinsics.unproject_depths(
            np.array([x, y]).T, depths[y, x]
        )
        points3d_colors = view.color[y, x]

        points3d_in_world = pose.apply(points3d)

        rr.log(f"/points/{i}", rr.Points3D(points3d_in_world, colors=points3d_colors), static=True)


    rr.set_time_seconds("stable_time", t)

    cam = Camera(
        vid.intrinsics,
        pose
    )
    rr.log("/camera_space", rr_utils.transform(pose))

    rr_utils.log_camera("/camera_space/camera", cam, view.color, view.depth_assumed)
