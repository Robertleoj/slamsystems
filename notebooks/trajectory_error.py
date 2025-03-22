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
from project.experimental.generate_trajectory import generate_random_trajectory
import rerun as rr
import project.utils.rerun_utils as rr_utils
from project.utils.spatial import Pose
from project.utils.image import Colors
import numpy as np

# %%
rr_utils.simple_init("app")
rr.log("/", rr.Clear(recursive=True))

# %%
N = 200
dt = 0.5
traj = generate_random_trajectory(N = N, dt=dt, rot_volatility=0.01, trans_volatility=0.1)

# %%
rr_utils.log_trajectory("/gt_traj", traj, Colors.RED)

# %%

# %%
disturbances = [
    Pose.exp(np.random.randn(6) * 0.01) for _ in range(N)
]
observed_traj = [d @ p for d, p in zip(disturbances, traj)]


# %%
rr_utils.log_trajectory("/seen_traj", observed_traj, Colors.GREEN)

# %%
sm = 0
for obs, gt in zip(observed_traj, traj):
    pose_diff = gt.inv @ obs
    diff_se3 = pose_diff.log()
    sm += (diff_se3 * diff_se3).sum()

rmse = np.sqrt(sm / N)
rmse


