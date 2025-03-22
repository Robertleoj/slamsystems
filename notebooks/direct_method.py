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

# %%
# %load_ext autoreload
# %autoreload 2
from project.camera_readers.realsense import RealSenseCamera
from project.utils.camera import View
from project.utils.spatial import Pose
from project.utils.camera.camera_params import Camera
from project.utils.image import to_greyscale
import project.utils.rerun_utils as rr_utils
import rerun as rr
import rerun.blueprint as rrb
import numpy as np
import cv2
from project.utils.drawing import draw_pixels, draw_lines
from project.utils.features import fast_detect
import matplotlib.pyplot as plt
from project.utils.optical_flow import lk_optical_flow
from project.utils.colors import Colors
import mediapy
from project.utils.paths import repo_root

# %% [markdown]
# # Direct method implementation
#
# 1. Get a view pair
# 2. Detect keypoints using FAST in the first one

# %%
base_path = repo_root() / "data" / "view_pairs" / "3"
view1 = View.load(base_path / "view1")
view2 = View.load(base_path / "view2")
mediapy.show_images([view1.color, view2.color])

# %%
fast = cv2.FastFeatureDetector.create()
fast.setNonmaxSuppression(True)
fast.setType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

# %%
grey1 = to_greyscale(view1.color)
detected_keypoints = fast_detect(grey1, fast)
len(detected_keypoints)

p1_unfiltered = np.array([kp.point.to_arr() for kp in detected_keypoints])
responses = np.array([kp.response for kp in detected_keypoints])
plt.hist(responses)
p1 = p1_unfiltered[np.argsort(responses)][-50:]


mediapy.show_image(draw_pixels(view1.color, p1, Colors.RED))

# %%
intrinsics = view1.intrinsics
im2_undistorted = intrinsics.undistort_image(view2.color)
grey2_undistorted = to_greyscale(im2_undistorted)
mediapy.show_images([view2.color, im2_undistorted, grey2_undistorted], columns=3)


# %%
p1 = p1.astype(int)

depths = view1.depth_assumed[
    p1[:, 1], p1[:, 0]
]
valid_depths = depths > 0

p1 = p1[valid_depths]
depths = depths[valid_depths]

P = intrinsics.unproject_depths(p1.astype(float), depths)

# %%
camera1 = Camera(
    intrinsics,
    Pose.identity()
)

rr_utils.init()
rr_utils.log_camera("/main/cam1", camera1, view1.color)
rr.log("/main/P", rr.Points3D(P))
rr.log("/main/cam1_arrows", rr.Arrows3D(vectors=P))

# %% [markdown]
# # Implement the direct method, single layer
#
# So, Symforce doesn't like evaluating images. Let's try using scipy.optimize instead.

# %%
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
import einops

# %%
camera_matrix = intrinsics.camera_matrix

# %%

factors = []


def error(
    x: np.ndarray, 
    p1: np.ndarray,
    P: np.ndarray, 
    camera_matrix: np.ndarray, 
    grey1: np.ndarray,
    grey2_undistorted: np.ndarray,
    patch_radius: int,
    *args,
    **kwargs
) -> np.ndarray:

    cam_pose = Pose.exp(x)

    # print(f"error cam pose: {cam_pose.rvec=}, {cam_pose.tvec=}")
    # N x 3
    P_cam2 = cam_pose.apply(P)

    N = P.shape[0]

    u = einops.einsum(camera_matrix,  P_cam2, "d1 d2, n d2 -> n d1")
    u /= u[:, 2, None]

    ux = np.round(u[:, 0]).astype(int)
    uy = np.round(u[:, 1]).astype(int)
    # print(u)

    px = np.round(p1[:, 0]).astype(int)
    py = np.round(p1[:, 1]).astype(int)

    I1_p1 = np.zeros((N,))
    I2_u = np.zeros((N,))

    for dx in range(-patch_radius, patch_radius + 1):
        for dy in range(-patch_radius, patch_radius + 1):

            I2_u += grey2_undistorted[uy + dy, ux + dx]
            I1_p1 += grey1[py + dy, px + dx]

    I1_p1 /= (patch_radius + 1) ** 2
    I2_u /= (patch_radius + 1) ** 2

    return I1_p1 - I2_u

def jacobian(x: np.ndarray, P: np.ndarray, camera_matrix: np.ndarray, grey2_undistorted: np.ndarray, patch_radius: int, *args, **kwargs):
    N = P.shape[0]

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]

    # jacobian is n x 6: n elements in the residual, 6 elements in the lie algebra
    J = np.zeros((N, 6))

    pose = Pose.exp(x)
    # print(f"Jacobian cam pose: {pose.rvec=}, {pose.tvec=}")

    P_cam2 = pose.apply(P)

    u = einops.einsum(camera_matrix,  P_cam2, "d1 d2, n d2 -> n d1")
    u /= u[:, 2, None]

    ux = np.round(u[:, 0]).astype(int)
    uy = np.round(u[:, 1]).astype(int)

    I2 = grey2_undistorted
    # shape N,
    dI2x_du = np.zeros((N,))
    dI2y_du = np.zeros((N,))

    for dx in range(-patch_radius, patch_radius + 1):
        for dy in range(-patch_radius, patch_radius + 1):
            dI2x_du += I2[uy + dy, ux + dx + 1] - I2[uy + dy, ux + dx - 1]
            dI2y_du += I2[uy+ dy + 1, ux + dx] - I2[uy + dy - 1, ux + dx]

    dI2x_du /= (patch_radius + 1) ** 2
    dI2y_du /= (patch_radius + 1) ** 2

    # Shape N x 2
    dI2_du = np.stack([dI2x_du, dI2y_du], axis=1)

    for i in range(N):
        Pi =  P[i]
        ui = u[i]

        X = Pi[0]
        Y = Pi[1]
        Z = Pi[2]

        du_dxi = np.array([
            [fx / Z, 0, -(fx * X) / (Z**2), - (fx * X * Y) / (Z ** 2), fx + (fx * X**2)/(Z**2), -(fx*Y)/Z],
            [0, fy / Z, -(fy * Y) / (Z**2), - fy - (fy * Y ** 2)/(Z**2), (fy * X * Y)/(Z ** 2), (fy*X)/Z]
        ])


        # print("du_dxi", du_dxi)
        
        dI2_dui = dI2_du[i].reshape(1, 2)

        # print("dI2_dui", dI2_dui)

        J[i, :] = dI2_dui @ du_dxi

    return -J

# %%
random_mask = np.random.random(p1.shape[0]) > 0.5

result = least_squares(
    error,
    np.zeros(6),
    jac=jacobian, # type: ignore
    kwargs=dict(
        P=P[random_mask],
        p1=p1[random_mask],
        camera_matrix=camera_matrix,
        grey2_undistorted=grey2_undistorted / 255.0,
        patch_radius=10,
        grey1=grey1 / 255.0
    ),
    # diff_step=1e-3,
    method="lm",
    xtol=1e-8
)

# %%
display(np.linalg.norm(result.jac))
display(result)


# %%
cam2_pose = Pose.exp(result.x)
camera2 = Camera(
    intrinsics,
    cam2_pose
)

# %%
rr_utils.log_camera("/main/cam2", camera2, view2.color)

# %%
result.cost

# %%
plt.plot(np.linalg.norm(result.jac, axis=1))

# %%
points_in_cam2 = cam2_pose.apply(P)
print(points_in_cam2.shape, points_in_cam2.dtype)
points_in_im2 = intrinsics.project_points(points_in_cam2.astype(np.float32))

drawn1 = draw_pixels(view1.color, p1, Colors.RED)
drawn2 = draw_pixels(view2.color, points_in_im2, Colors.RED)

mediapy.show_images([drawn1, drawn2])

# %%
cam2_pose.tvec, cam2_pose.rvec
