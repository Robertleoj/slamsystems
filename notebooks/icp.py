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
# # Implementing ICP
# Steps:
# 1. Match features in two views
# 2. Get 3D position from view 1, get PnP problem
# 3. Implement PnP with Ceres

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from project.utils.spatial import Pose, Rotation
from project.utils.camera import View, Camera
from project.utils.colors import Colors
from project.utils.paths import repo_root
from project.utils.features import get_flann_matcher_for_orb, orb_feature_detect_and_match, draw_matches
import cv2
import einops
import numpy as np
import mediapy
import rerun as rr
import rerun.blueprint as rrb
import project.utils.rerun_utils as rr_utils
from project.foundation.reinventing import icp_g2o, icp_ceres

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
                origin="/icp"
            ),
            rrb.Spatial3DView(
                origin="/centered"
            ),
            rrb.Spatial3DView(
                origin="/aligned"
            ),
            rrb.Spatial3DView(
                origin="/g2o_aligned"
            ),
            rrb.Spatial3DView(
                origin="/ceres_aligned"
            ),
            rrb.Spatial3DView(
                origin="/pypose_aligned"
            ),
            rrb.Spatial3DView(
                origin="/symforce_aligned"
            )
        ),
        collapse_panels=True
    ),
)

# %%
base_path = repo_root() / "data/two_views_2"
view1 = View.load(base_path / 'view1')
rr.log("/images/view1", rr.Image(view1.color))
view2 = View.load(base_path / 'view2')
rr.log("/images/view2", rr.Image(view2.color))

# %%
mediapy.show_images([view1.color, view2.color])

# %%
print(view1.depth_assumed.dtype)

# %%
orb = cv2.ORB.create()
flann = get_flann_matcher_for_orb()

match_result = orb_feature_detect_and_match(view1.color, view2.color, orb, flann)
drawn_result = draw_matches(
    view1.color, match_result.keypoints_1, view2.color, match_result.keypoints_2, match_result.matches
)
mediapy.show_image(drawn_result)
rr.log("/images/matched_views", rr.Image(drawn_result))

# %% [markdown]
# # Implement ICP

# %%

# %%

matched_pixels1_depths = view1.depth_assumed[
    match_result.matched_points_1[:, 1].astype(int),
    match_result.matched_points_1[:, 0].astype(int)
]


matched_pixels2_depths = view2.depth_assumed[
    match_result.matched_points_2[:, 1].astype(int),
    match_result.matched_points_2[:, 0].astype(int)
]

valid_depth_mask = (matched_pixels1_depths > 0) & (matched_pixels2_depths > 0)

matched_pixels1_depths = matched_pixels1_depths[valid_depth_mask]
matched_pixels1 = match_result.matched_points_1[valid_depth_mask]

matched_pixels2_depths = matched_pixels2_depths[valid_depth_mask]
matched_pixels2 = match_result.matched_points_2[valid_depth_mask]

# %%
intrinsics = view1.intrinsics

# %%
points1 = intrinsics.unproject_depths(matched_pixels1, matched_pixels1_depths)
points2 = intrinsics.unproject_depths(matched_pixels2, matched_pixels2_depths)

# %%
rr.log("/icp/origin", rr_utils.triad())
rr_utils.log_point_matches("/icp/point_matches", points1, points2)

# %%
centroid1 = np.mean(points1, axis=0)
centroid2 = np.mean(points2, axis=0)

rr.log("/icp/centroids", rr.Points3D([centroid1, centroid2], colors=[Colors.CERISE_PINK, Colors.CONIFER], radii=7.0))

# %%
decentroids1 = points1 - centroid1
decentroids2 = points2 - centroid2

# %%
rr.log("/centered/origin", rr_utils.triad())
rr_utils.log_point_matches("/centered", decentroids1, decentroids2)

# %%
W = einops.einsum(decentroids1, decentroids2, "n d1, n d2 -> d1 d2")
W

# %%
U, S, Vs = np.linalg.svd(W)
U, S, Vs

# %%
R = U @ Vs
print(np.linalg.det(R))

# %%
t = centroid1 - R @ centroid2
t

# %%
camera2_pose = Pose.from_rotmat_trans(R, t)

# %%
camera1 = Camera(
    intrinsics,
    Pose.identity()
)

camera2 = Camera(
    intrinsics,
    camera2_pose
)

# %%
rr_utils.log_camera("/aligned/camera1", camera1)
rr_utils.log_camera("/aligned/camera2", camera2)

# %%
rr.log("/aligned/points1", rr.Points3D(points1, radii=3.0))

# %%
points2_aligned = camera2_pose.apply(points2)

# %%
rr.log("/aligned/points2_aligned", rr.Points3D(points2_aligned, radii=3.0))

# %%
rr_utils.log_point_matches("/aligned/point_matches", points1, points2_aligned)

# %% [markdown]
# # Solve with g2o

# %%
initial_guess = Pose.identity()
cam2_pose_se3_g2o = icp_g2o(points1, points2, initial_guess.rvec, initial_guess.tvec)
display(cam2_pose_se3_g2o)

cam2_pose_g2o = Pose.exp(cam2_pose_se3_g2o)

# %%
cam2_g2o = Camera(
    intrinsics,
    cam2_pose_g2o
)

# %%
rr.log("/g2o_aligned/origin", rr_utils.triad())

# %%
points2_aligned_g2o = cam2_pose_g2o.apply(points2)

# %%
rr_utils.log_camera("/g2o_aligned/cam1", camera1)
rr_utils.log_camera("/g2o_aligned/cam2", cam2_g2o)

# %%
rr_utils.log_point_matches("/g2o_aligned/matches", points1, points2_aligned_g2o)

# %% [markdown]
# # Solve with Ceres

# %%
initial_guess = Pose.identity()
cam2_pose_se3_ceres = icp_ceres(points1, points2, initial_guess.rvec, initial_guess.tvec)
display(cam2_pose_se3_ceres)

cam2_pose_ceres = Pose.exp(cam2_pose_se3_ceres)

cam2_ceres = Camera(
    intrinsics,
    cam2_pose_ceres
)
rr.log("/ceres_aligned/origin", rr_utils.triad())
points2_aligned_ceres = cam2_pose_ceres.apply(points2)

rr_utils.log_camera("/ceres_aligned/cam1", camera1)
rr_utils.log_camera("/ceres_aligned/cam2", cam2_ceres)
rr_utils.log_point_matches("/ceres_aligned/matches", points1, points2_aligned_ceres)

# %% [markdown]
# # Try pypose

# %%
from project.utils.pointclouds.align import align_point_matches_pypose

# %%
cam_pose_pypose = align_point_matches_pypose(points1, points2)

# %%

rr_utils.log_camera("/pypose_aligned/cam1_pypose",camera1)
rr_utils.log_camera("/pypose_aligned/cam2_pypose", Camera(intrinsics, cam_pose_pypose))

# %%
points2_aligned_pypose = cam_pose_pypose.apply(points2)

# %%
rr_utils.log_point_matches("/pypose_aligned/matches", points1, points2_aligned_pypose)

# %% [markdown]
#
# pypose sucks, but I finally got symforce working!
# # Symforce

# %%
import symforce
if not 'epsilon_set' in globals():
    symforce.set_epsilon_to_symbol()
    epsilon_set = True


import symforce.symbolic as sf
from symforce.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from project.utils.symforce_utils import pose_to_sf, sf_to_pose

# %%
initial_values = Values(
    pose=pose_to_sf(Pose.identity())
)

# %%
factors = []
for point1, point2 in zip(points1, points2):
    p1 = sf.V3(*point1.squeeze())
    p2 = sf.V3(*point2.squeeze())

    def factor(pose: sf.Pose3) -> sf.V3:
        return p1 - (pose * p2) # type: ignore

    factors.append(Factor(
        residual=factor,
        keys=["pose"]
    ))
    

# %%
params = Optimizer.Params(
    debug_stats=True
)
optimizer = Optimizer(
    factors=factors,
    optimized_keys=['pose'],
    params=params
)

# %%
result = optimizer.optimize(initial_values)

# %%
result

# %%
cam2_pose_sf = sf_to_pose(result.optimized_values['pose'])
cam2_pose_sf

# %%
cam2_sf = Camera(
    intrinsics,
    cam2_pose_sf
)

# %%
points2_aligned = cam2_pose_sf.apply(points2)

# %%
rr_utils.log_camera("/symforce_aligned/cam1", camera1)
rr_utils.log_camera("/symforce_aligned/cam2", cam2_sf)

# %%
rr_utils.log_point_matches("/symforce_aligned/aligned", points1, points2_aligned)
