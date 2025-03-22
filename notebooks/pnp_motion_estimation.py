# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: project-ePUsKrUH-py3.10
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Estimation of camera motion with PnP
#
# 1. Get point matches with ORB feature matching
# 2. Find 3D locations of points in camera 1 using depth map
# 3. Solve position of camera 2 using PnP on the 3D points and their known pixel locations in camera2

# %%
from project.utils.camera import View, Camera
from project.utils.spatial import Pose
from pathlib import Path
from project.utils.features import get_flann_matcher_for_orb, orb_feature_detect_and_match, draw_matches
from project.visual_odometry.motion_estimation import estimate_motion_from_matches
from project.utils.viser_utils import show_camera, show_line_segments, show_points, show_projection_rays
from project.utils.image import draw_pixels, Color, draw_lines
import matplotlib.pyplot as plt
import viser
import cv2
import mediapy
import numpy as np

# %%
base_path= Path("../data/two_views")
view1 = View.load(base_path / 'view1')
view2 = View.load(base_path / 'view2')

# %%
display(view1.depth_assumed.mean(), view1.depth_assumed.max(), view1.depth_assumed.dtype)
display(view2.depth_assumed.mean(), view2.depth_assumed.max(), view2.depth_assumed.dtype)

# %%
mediapy.show_images([view1.color, view1.depth_assumed, view2.color, view2.depth_assumed], titles=["color1", "depth1", "color2", "depth2"], columns=2)

# %% [markdown]
# ## Get point matches

# %%
orb = cv2.ORB.create()
flann = get_flann_matcher_for_orb()

# %%
match_result = orb_feature_detect_and_match(
    view1.color, view2.color, orb, flann
)

# %%
drawn = draw_matches(
    view1.color, match_result.keypoints_1, view2.color, match_result.keypoints_2, match_result.matches
)
mediapy.show_image(drawn)

# %% [markdown]
# ## Get 3D coordinates of keypoints in camera 1

# %%
if 'vis' not in globals():
    vis = viser.ViserServer()
vis.scene.reset()

# %%
intrinsics = view1.intrinsics

# %%
camera1 = Camera(
    intrinsics=intrinsics,
    extrinsics=Pose.identity()
)

# %%
show_camera(vis, camera1, name='cam1', image=view1.color)

# %%
cam1_pixels = match_result.matched_points_1
cam2_pixels = match_result.matched_points_2

# %%
cam1_pixel_depths = view1.depth_assumed[cam1_pixels[:, 1].astype(int), cam1_pixels[:, 0].astype(int)]

# %%
valid_depths_mask = cam1_pixel_depths > 0
valid_depths_mask.sum() / valid_depths_mask.shape[0]

# %%
cam1_pixels = cam1_pixels[valid_depths_mask]
cam2_pixels = cam2_pixels[valid_depths_mask]
cam1_pixel_depths = cam1_pixel_depths[valid_depths_mask]

# %%
cam1_pixel_depths.shape

# %%
world_points = intrinsics.unrpoject_depths(cam1_pixels, cam1_pixel_depths)
world_points.shape

# %%
show_projection_rays(vis, world_points, name="cam1_projection_rays")
show_points(vis, world_points, name="world_points", point_size=0.02)

# %%
debug_img = view2.color.copy()
world_to_cam2, repr_errors, inlier_mask = intrinsics.estimate_pose_pnp_ransac(world_points, cam2_pixels, world_to_cam_initial_guess=Pose.identity(), debug_img=debug_img)

# %%
mediapy.show_image(debug_img)

# %%
print(f"num inliers {inlier_mask.sum()}/{inlier_mask.shape[0]}")

# %%
camera2 = Camera(
    intrinsics=intrinsics,
    extrinsics=world_to_cam2
)

# %%
show_camera(vis, camera2, name='cam2', image=view2.color)

# %%
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].hist(repr_errors)
axes[1].hist(repr_errors[inlier_mask])

# %%
print(repr_errors.mean(), repr_errors[inlier_mask].mean())

# %%
colors = np.zeros((world_points.shape[0], 3))
colors[inlier_mask] = (0, 255, 0)
colors[~inlier_mask] = (255, 0, 0)

show_points(vis, world_points, colors=colors, name="world_points", point_size=0.02)


# %%
show_projection_rays(vis, camera2.extrinsics.inv.apply(world_points), name="cam2_proj", camera_pose=camera2.extrinsics)
