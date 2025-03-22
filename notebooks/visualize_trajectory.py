# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
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
import viser
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from project.utils.spatial.pose import Pose

# %%
if 'vis' not in globals():
    vis = viser.ViserServer()


# %%
@dataclass(frozen=True)
class State:
    pose: Pose
    angular_velocity: np.ndarray
    translational_velocity: np.ndarray


# %%
initial_state = State(
    pose=Pose.identity(),
    angular_velocity=np.array([0, 0, 0]),
    translational_velocity=np.array([0, 0, 0])
)

states = [initial_state]


# %%
dt = 0.5
curr_state = initial_state


# %%
def update_state(state: State, dt: float) -> State:
    dR = R.from_rotvec(state.angular_velocity * dt).as_matrix()
    new_R = dR @ state.pose.rot_mat

    new_t = state.pose.tvec + dt * state.translational_velocity

    new_pose = Pose.from_rotmat_trans(new_R, new_t)

    new_angular_velocity = state.angular_velocity + np.random.randn(*state.angular_velocity.shape) * 0.1 * dt
    new_translational_velocity = state.translational_velocity + np.random.randn(*state.translational_velocity.shape) * 0.1 * dt

    return State(
        pose=new_pose,
        angular_velocity=new_angular_velocity,
        translational_velocity=new_translational_velocity
    )
    


# %%
for i in range(1000):
    new_state = update_state(curr_state, dt)
    states.append(new_state)
    curr_state = new_state

# %%
vis.scene.reset()

# %%


for i, s in enumerate(states):
    vis.scene.add_frame(
        f"/traj/sample_{i}",
        show_axes=True,
        position=s.pose.tvec,
        wxyz=s.pose.wxyz
    )

vis.flush()


