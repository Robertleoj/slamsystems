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
# # PnP implementation with Ceres
# Steps:
# 1. Match features in two views
# 2. Get 3D position from view 1, get PnP problem
# 3. Implement PnP with Ceres

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from project.utils.spatial import Pose
from project.utils.camera import View, Camera
from project.utils.image import Colors
from project.utils.features import get_flann_matcher_for_orb, orb_feature_detect_and_match, draw_matches
from project.foundation.reinventing import solve_pnp_ceres, solve_pnp_g2o
import cv2
import numpy as np
from pathlib import Path
import mediapy
import rerun as rr
import rerun.blueprint as rrb
import project.utils.rerun_utils as rr_utils

# %%
rr_utils.init("pnp")
rr.log("/", rr.Clear(recursive=True))
rr.send_blueprint(
    rrb.Blueprint(
        rrb.Tabs(
            rr_utils.image_tabs_bp(
                "images", ["view1", "view2", "matched_views"]
            ),
            rrb.Spatial3DView(
                origin="/pnp"
            )
        ),
        collapse_panels=True
    ),
)

# %%
base_path = Path("../data/two_views")
view1 = View.load(base_path / 'view1')
rr.log("/images/view1", rr.Image(view1.color))
view2 = View.load(base_path / 'view2')
rr.log("/images/view2", rr.Image(view2.color))

# %%
mediapy.show_images([view1.color, view2.color])

# %%
orb = cv2.ORB.create()
flann = get_flann_matcher_for_orb()

match_result = orb_feature_detect_and_match(view1.color, view2.color, orb, flann)
drawn_result = draw_matches(
    view1.color, match_result.keypoints_1, view2.color, match_result.keypoints_2, match_result.matches
)
mediapy.show_image(drawn_result)
rr.log("/images/matched_views", rr.Image(drawn_result))

# %%

matched_pixels1_depths = view1.depth_assumed[
    match_result.matched_points_1[:, 1].astype(int),
    match_result.matched_points_1[:, 0].astype(int)
]

valid_depth_mask = matched_pixels1_depths > 0

matched_pixels1_depths = matched_pixels1_depths[valid_depth_mask]
matched_pixels1 = match_result.matched_points_1[valid_depth_mask]
matched_pixels2 = match_result.matched_points_2[valid_depth_mask]

# %%
intrinsics = view1.intrinsics

# %%
world_points = intrinsics.unproject_depths(matched_pixels1, matched_pixels1_depths)

# %%
rr.log("/pnp/world_points", rr.Points3D(world_points, radii=3.0, colors=Colors.RED))
rr.log("/pnp/cam1_rayw", rr.Arrows3D(
    vectors=world_points
))

# %%
camera1 = Camera(
    intrinsics,
    Pose.identity()
)

rr_utils.log_camera(
    "/pnp/cam1",
    cam=camera1,
    image=view1.color,
    depth_image=view1.depth_assumed,
    point_fill_ratio=0.5
)

# %% [markdown]
# # Implement pnp

# %%
matched_pixels2_undistorted = intrinsics.undistort_pixels(matched_pixels2)

# %%
print(matched_pixels2_undistorted.dtype)

# %%
initial_guess = Pose.identity()

img_points = [
    mp.astype(np.float64).reshape(2, 1).copy()
    for mp in matched_pixels2_undistorted
]

obj_points = [
    wp.astype(np.float64).reshape(3, 1).copy()
    for wp in world_points
]
print(world_points.shape)

rvec, tvec = solve_pnp_g2o(
    matched_pixels2_undistorted,
    world_points,
    intrinsics.camera_matrix,
    initial_guess.rvec,
    initial_guess.tvec
)

# %%
rvec, tvec

# %%
cam_pose = Pose.from_rvec_tvec(rvec, tvec).inv
camera2 = Camera(
    intrinsics,
    cam_pose
)
rr_utils.log_camera(
    "/pnp/cam2", 
    camera2, 
    view2.color,
    view2.depth_assumed,
    point_fill_ratio=0.5
)


# %%
matched_pixels2_depths = view2.depth_assumed[
    matched_pixels2[:, 1].astype(int),
    matched_pixels2[:, 0].astype(int)
]

points_in_cam_2 = camera2.intrinsics.unproject_depths(
    matched_pixels2,
    matched_pixels2_depths
)

rr.log("/pnp/cam2/rays", rr.Arrows3D(
    vectors=points_in_cam_2,
))

# %%

cam_pose_opencv, repr_errors = intrinsics.estimate_pose_pnp(
    world_points,
    matched_pixels2,
    Pose.identity(),
)

# %%
cam_pose_opencv.inv.rvec, cam_pose_opencv.inv.tvec

# %%
projected = camera2.project_points(world_points)
