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
# # Plantage shed bundle adjustment from scratch
#
# Here I'll do bundle adjustment completely from scratch, on the plantage shed video. 

# %%
# %load_ext autoreload
# %autoreload 2
import symforce
if 'eps_set' not in globals():
    symforce.set_epsilon_to_symbol()
    eps_set = True


import json
import cv2
import einops
from project.utils.camera import Intrinsics, Camera
from project.utils.markers import DetectedTag
from project.utils.spatial import Pose
from project.utils.markers import get_corners_in_tag
from pathlib import Path
from project.utils.paths import repo_root, symforce_codegen_path
import project.utils.rerun_utils as rr_utils
import rerun as rr
import dt_apriltags as april
from typing import cast
from project.utils.symforce_utils import pose_to_sf
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import imageio.v3 as iio
from project.utils.image import to_greyscale
import random
import mediapy
from project.utils.paths import repo_root
from project.tag_slam.epipolar_line_search import TagSlam

# %%
TAG_SIDE_LENGTH = 42

# %%
base_path = repo_root() / "data/monumental_take_home"
camera_json = json.loads((base_path / 'cam.json').read_text())
intrinsics = Intrinsics.from_pinhole_params(
    camera_json['width'],
    camera_json['height'],
    camera_json['fx'],
    camera_json['fy'],
    camera_json['px'],
    camera_json['py'],
    np.array(camera_json['dist_coeffs'])
)

# %%
rr_utils.init("tagslam_with_ba")

# %%
vid = iio.imiter(base_path / 'plantage_shed.mp4')

# %%
tagslam = TagSlam(TAG_SIDE_LENGTH, intrinsics, estimate_dense=True, keyframe_frequency=50)


# %%
def log_state(path: str, tagslam: TagSlam):

    rr.log(f"{path}/camera_trajectory/path", rr.LineStrips3D([p.tvec for p in tagslam.camera_poses]))

    rr.log(f"{path}/camera_trajectory/start", rr.Points3D(tagslam.camera_poses[0].tvec, radii=5.0))

    for tag_id, tag in tagslam.landmarks.items():
        rr.log(f"{path}/tags/tag{tag_id}", rr_utils.transform(tag.pose))
        rr.log(f"{path}/tags/tag{tag_id}/triad", rr_utils.triad())

    for keyframe_idx in tagslam.keyframe_indices:
        cam_pose = tagslam.camera_poses[keyframe_idx]

        camera = Camera(
            tagslam.undistorted_intrinsics,
            cam_pose
        )

        rr_utils.log_camera(f"{path}/cams/cam{keyframe_idx}", camera)

        if tagslam.keyframe_depth_estimators is not None:
            dense_map = tagslam.keyframe_depth_estimators[keyframe_idx].get_colored_points_in_world()
            if dense_map is not None:
                points3d, colors = dense_map

                rr.log(f"{path}/pointclouds/pointcloud{keyframe_idx}", rr.Points3D(points3d, colors=colors))



# %%
depth_map_frames = []
for i, frame in tqdm(enumerate(vid)):
    tagslam.process_frame(frame)

    if i % 10 == 0:
        rr.set_time_sequence("slamming", i)
        log_state("/live_slam", tagslam)


    

# %%
tagslam.refine()

# %%
log_state("/refined", tagslam)

# %%
from timeit import default_timer
import matplotlib.pyplot as plt

from project.utils.colors import Colors
from project.utils.drawing import draw_pixels
from project.utils.triangulation import check_epipolar_constraint, project_onto_epipolar_line, triangulate_normalized_image_points
kf1_idx, kf2_idx = tagslam.keyframe_indices[:2]

frame1_distorted = tagslam.keyframe_imgs[kf1_idx]
frame2_distorted = tagslam.keyframe_imgs[kf2_idx]

cam1_pose = tagslam.camera_poses[kf1_idx]
cam2_pose = tagslam.camera_poses[kf2_idx]


# %%

# %%

# Convert frames to grayscale
from traitlets import default
from project.utils.depth_estimation.block_search import epipolar_line_search_block_match

frame1 = intrinsics.undistort_image(frame1_distorted)
frame2 = intrinsics.undistort_image(frame2_distorted)

grey1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
grey2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

undistorted_intrinsics = intrinsics.to_undistorted()

# Compute dense optical flow using Farneback

# Initialize DeepFlow
start = default_timer()

# Deepflow
# deepflow = cv2.optflow.createOptFlow_DeepFlow()
# flow = deepflow.calc(grey1, grey2, None)

# PCAFlow
# pcaflow = cv2.optflow.createOptFlow_PCAFlow()
# flow = pcaflow.calc(gray1, gray2, None)

# flow = cv2.calcOpticalFlowFarneback(grey1, grey2, None, 0.5, 5, 30, 10, 5, 1.2, 0)

height,width = frame1.shape[:2]

x_to_match = np.linspace(0, width-1, 200)
y_to_match = np.linspace(0, height-1, 150)

# Create a grid of coordinates
X, Y = np.meshgrid(x_to_match, y_to_match)

# Stack them into a list of coordinates
points = np.vstack([X.ravel(), Y.ravel()]).T.astype(np.int64)

start = default_timer()
# using line search
matched_pixels, valid_mask = epipolar_line_search_block_match(points, grey1, grey2, cam1_pose.inv @ cam2_pose, undistorted_intrinsics, undistort=False, search_zone_size=250, block_size=5)
end = default_timer()
print(f"Line search took {end - start}")


# %%

# y, x = np.where(np.linalg.norm(matched_pixels, axis=-1) > 1.0)

pixels1 = points[valid_mask.astype(bool)]
pixels2 = matched_pixels[valid_mask.astype(bool)]


# end = default_timer()
# print(f"time: {end - start}")

# y, x = np.where(np.linalg.norm(flow, axis=-1) > 1.0)

# pixels1 = np.array([x, y]).T
# print(pixels1.shape)
# pixels2 = pixels1 + flow[y, x]
# print(pixels2.shape)


# %%
pixels1

# %%

pixels1_normalized = undistorted_intrinsics.normalize_pixels(pixels1)
pixels2_normalized = undistorted_intrinsics.normalize_pixels(pixels2)

valid= check_epipolar_constraint(pixels1_normalized, pixels2_normalized, (cam1_pose.inv @ cam2_pose).inv, 1e-1)

print(f"{sum(valid)}/{valid.shape[0]} valid points under epipolar constraint")

pixels1_normalized = pixels1_normalized[valid]
pixels2_normalized = pixels2_normalized[valid]

# Apply RANSAC to filter out outliers
F, mask = cv2.findFundamentalMat(pixels1_normalized, pixels2_normalized, cv2.FM_RANSAC, 3.0, 0.99)
mask = mask.ravel().astype(bool)

pixels1_normalized = pixels1_normalized[mask]
pixels2_normalized = pixels2_normalized[mask]

points3d= triangulate_normalized_image_points([pixels1_normalized, pixels2_normalized], [cam1_pose, cam2_pose])

points3d_in_cam1 = cam1_pose.apply(points3d)
points3d = points3d[points3d_in_cam1[:, 2] > 0]


# project back into cam1 to get colors
proj = undistorted_intrinsics.project_points(points3d, cam1_pose)

x, y = proj.T.astype(int)
valid = (x > 0) & (x < frame1.shape[1]) & (y > 0) & (y < frame.shape[0])
x = x[valid]
y = y[valid]
proj = proj[valid]
points3d = points3d[valid]

pixel_values = frame1[y, x]
rr.log("/dense/points1", rr.Points3D(points3d, colors=pixel_values))

# Draw flow vectors
step = 16  # Grid step size
h, w = grey1.shape

random_indices = np.random.choice(len(pixels1), 5, replace=False)
pix1_draw = pixels1[random_indices]
pix2_draw = pixels2[random_indices]


db1 =draw_pixels(frame1, pix1_draw, Colors.RED, radius=5)
db2 =draw_pixels(frame2, pix2_draw, Colors.RED, radius=5)

# Draw filtered vectors
# step = 250
# vis = cv2.cvtColor(grey1, cv2.COLOR_GRAY2BGR)

# for y in range(0, grey1.shape[0], step):
#     for x in range(0, grey1.shape[1], step):
#         px, py = matched_pixels[y, x]
#         if px != 0 and py != 0:  # Only draw valid vectors
#             end_x, end_y = int(px), int(py)
#             cv2.arrowedLine(vis, (x, y), (end_x, end_y), (0, 255, 0), 5, tipLength=0.3)


mediapy.show_images([frame1, frame2, grey1, grey2, db1, db2], columns=2)


# %%
