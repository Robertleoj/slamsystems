# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: project-ePUsKrUH-py3.10
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
from project.camera_readers.realsense import RealSenseCamera
import mediapy
from project.utils.spatial.pose import rz_R, Pose
from project.utils.features import orb_detect, orb_detect_and_compute, draw_keypoints, match_feature_descriptors, draw_matches, get_flann_matcher_for_orb, get_good_matches
import numpy as np
from scipy.linalg import null_space
from itertools import combinations
import viser
import matplotlib.pyplot as plt
import cv2
import logging
from tqdm import trange

# %%
if 'cam' not in globals():
    cam = RealSenseCamera()

# %%
orb = cv2.ORB.create()
flann = get_flann_matcher_for_orb()

# %%
if 'vis' not in globals():
    vis: viser.ViserServer = viser.ViserServer()


# %%
from project.utils.viser_utils import show_pose
from project.visual_odometry.orb_basic import estimate_camera_movement


trajectory: list[Pose] = []
movements: list[Pose] = []

curr_pose = Pose.identity()
curr_view = cam.get_view()

_, fov = curr_view.intrinsics.fov()
aspect = curr_view.intrinsics.aspect_ratio()

trajectory.append(curr_pose)

vis.scene.reset()

for i in trange(50):
    next_view = cam.get_view()
    R, t = estimate_camera_movement(curr_view, next_view, orb, flann)

    cam_1_R_cam_2 = R.T
    cam_1_t_cam_2 = - (R.T @ (t.reshape(3, 1))).reshape(3)
    cam_1_T_cam_2 = Pose.from_rotmat_trans(cam_1_R_cam_2, cam_1_t_cam_2)

    curr_view = next_view
    curr_pose = curr_pose @ cam_1_T_cam_2

    trajectory.append(curr_pose)
    movements.append(cam_1_T_cam_2)


# %%
def plot_movement(diffs):
    rot_vecs = np.array([p.scipy_rot.as_euler('xyz', degrees=True) for p in diffs])

    fig, axes = plt.subplots(4, 3, figsize=(20, 20))

    axes[0, 0].hist(rot_vecs[:,0])
    axes[0, 1].hist(rot_vecs[:,1])
    axes[0, 2].hist(rot_vecs[:,2])

    axes[1, 0].plot(rot_vecs[:,0])
    axes[1, 1].plot(rot_vecs[:,1])
    axes[1, 2].plot(rot_vecs[:,2])

    translations = np.array([p.translation for p in diffs])

    axes[2, 0].hist(translations[:,0])
    axes[2, 0].set_title("x")
    axes[2, 1].hist(translations[:,1])
    axes[2, 2].set_title("y")
    axes[2, 2].hist(translations[:,2])
    axes[2, 2].set_title("z")

    axes[3, 0].plot(translations[:,0])
    axes[3, 0].set_title("x")
    axes[3, 1].plot(translations[:,1])
    axes[3, 1].set_title("y")
    axes[3, 2].plot(translations[:,2])
    axes[3, 2].set_title("z")

    fig.tight_layout()

    plt.show()
plot_movement(movements)

# %%
import numpy as np
from scipy.spatial.transform import Rotation as R

def smooth_pose_diffs(pose_diffs: list[Pose], window_size=3):
    # Extract translation and rotation (axis-angle) from Pose3 diffs
    trans = np.array([pose.tvec for pose in pose_diffs])
    rots = np.array([pose.rvec for pose in pose_diffs])  # SO(3) -> Lie algebra (axis-angle)

    # Median filter translations and rotations independently
    trans_smooth = np.array([np.median(trans[i:i + window_size], axis=0) for i in range(len(trans) - window_size + 1)])
    rots_smooth = np.array([np.median(rots[i:i + window_size], axis=0) for i in range(len(rots) - window_size + 1)])

    return [
        Pose.from_rvec_tvec(rotvec, trans) for rotvec, trans in zip(rots_smooth, trans_smooth)
    ]


def make_trajectory(pose_diffs: list[Pose]):
    curr_smooth_pose = Pose.identity()
    
    new_trajectory = [curr_smooth_pose]

    for diff in pose_diffs:
        curr_smooth_pose = curr_smooth_pose @ diff
        new_trajectory.append(curr_smooth_pose)

    return new_trajectory


# %%

smooth_movements = smooth_pose_diffs(movements)
plot_movement(smooth_movements)

smooth_trajectory = make_trajectory(smooth_movements)


for i, pose in enumerate(smooth_trajectory):
    scene_item = f'cam/{i}'
    show_pose(vis, pose, name=scene_item)

    vis.scene.add_camera_frustum(
        name=f"{scene_item}/frust",
        fov=fov,
        aspect=aspect
    )

# %%
from project.utils.viser_utils import show_line_segments


translations = np.array([p.tvec for p in smooth_trajectory])
starts = translations[:-1]
ends = translations[1:]

show_line_segments(vis, starts, ends, name="traj", thickness=5)


# %%
from project.utils.viser_utils import show_points


show_points(vis, np.zeros((1, 3)), name="origin")

# %%
vis2 = viser.ViserServer()

# %%
vis2.scene.add_frame('Origin')

# %%
vis2.scene.add_line_segments(
    name="line",
    points=np.array([[
        [1, 0, 0],
        [2, 0, 0]
    ]]),
    colors=(255, 0, 0)
)

# %%
vis2.scene.add_frame(
    name='test_pose',
    position=np.array([-1, 0, 0])
)

# %%
plot_movement(smooth_trajectory)
