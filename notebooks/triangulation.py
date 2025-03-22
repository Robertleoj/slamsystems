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
from project.utils.camera.view import View
from pathlib import Path
import mediapy
import numpy as np
import einops
import cv2
from project.utils.features import get_flann_matcher_for_orb
from project.utils.features import orb_feature_detect_and_match
from project.utils.triangulation import triangulate_points
from project.utils.camera.camera_params import Camera
from project.utils.features import draw_matches
from project.utils.spatial.pose import Pose
import viser
from project.utils.viser_utils import show_pose, show_line_segments, show_points, show_camera
from project.utils.triangulation import triangulate_normalized_image_points, triangulate_pixels

# %%
base_path = Path("../data/two_views")
view1 = View.load(base_path / 'view1')
view2 = View.load(base_path / 'view2')

# %%
mediapy.show_images([view1.color, view2.color])

# %%


orb = cv2.ORB.create()
flann = get_flann_matcher_for_orb()

match_result = orb_feature_detect_and_match(
    view1.color, view2.color, orb, flann
)

# %%
mediapy.show_image(
    draw_matches(
        view1.color, 
        match_result.keypoints_1, 
        view2.color, 
        match_result.keypoints_2, 
        match_result.matches
    )
)

# %%
from project.visual_odometry.motion_estimation import estimate_motion_from_matches


R, t = estimate_motion_from_matches(
    match_result.matched_points_1,
    match_result.matched_points_2,
    view1.intrinsics
)

# %%
R, t 

# %%
cam_2_in_cam_1 = Pose.from_rotmat_trans(R, t.squeeze()).inv

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

camera2 = Camera(
    intrinsics=intrinsics,
    extrinsics=cam_2_in_cam_1
)

# %%
show_camera(vis, camera1, name='cam1')
show_camera(vis, camera2, name='cam2')

# %%
debug_dict = {}
triangulated_points = triangulate_pixels(
    [match_result.matched_points_1, match_result.matched_points_2],
    [camera1, camera2],
    debug_dict
)

# %%
rays = debug_dict['rays']

# %%
num_rays_to_draw = 20
rays_to_draw = rays[:num_rays_to_draw]

colors = np.random.randint(0, 255, (rays_to_draw.shape[0],2, 3))
print(colors.shape)


lengths = 10
cam1_start = rays_to_draw[:, 0, 0, :]
cam1_ends = rays_to_draw[:, 0, 1, :] * lengths

cam2_start = rays_to_draw[:, 1, 0, :]
cam2_ends = (rays_to_draw[:, 1, 1, :] - cam2_start) * lengths + cam2_start
show_line_segments(
    vis, cam1_start, cam1_ends, thickness=2, name='cam1_lines', colors=colors
)
show_line_segments(
    vis, cam2_start, cam2_ends, thickness=2, name='cam2_lines', colors=colors
)


# %%
show_points(vis, triangulated_points[:num_rays_to_draw], name='triangulated')
