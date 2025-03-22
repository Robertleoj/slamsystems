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

# %% [markdown] vscode={"languageId": "plaintext"}
# ## Imports

# %%
from project.camera_readers.realsense import RealSenseCamera
from project.utils.camera import Video
from tqdm import tqdm
from project.utils.paths import repo_root
import numpy as np

# %% [markdown]
# ## Capture video

# %%
cam = RealSenseCamera()

# %%
intrinsics = cam.get_intrinsics()

vid_color: list[np.ndarray] = []
vid_depth: list[np.ndarray] = []

# %%
num_frames = 300

# %%

for _ in tqdm(range(num_frames)):
    view = cam.get_view()
    color = view.color
    depth = view.depth_assumed

    vid_color.append(color)
    vid_depth.append(depth)

# %%
video = Video(
    color=vid_color,
    intrinsics=intrinsics,
    depth=vid_depth
)

# %%
video.save(repo_root() / 'data/videos/vid2')
