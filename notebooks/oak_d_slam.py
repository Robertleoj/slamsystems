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
from project.oak_slam.pipeline import OakReader
import project.utils.rerun_utils as rr_utils
import rerun as rr
from project.foundation.oak_slam import OakSlam
from project.foundation import set_spdlog_level
import mediapy

# %%
set_spdlog_level("debug")

# %%
reader = OakReader()

# %%
reader.color_intrinsics.distortion_parameters

# %%
slam = OakSlam(
    reader.color_intrinsics.to_cpp(),
    reader.left_intrinsics.to_cpp(),
    reader.right_intrinsics.to_cpp(),
    reader.center_to_left.mat,
    reader.center_to_right.mat
)

# %%

while True:
    frame = reader.get_frame()
    slam.process_frame(
        frame.center_color.img,
        frame.left_mono.img,
        frame.right_mono.img
    )

# %%
