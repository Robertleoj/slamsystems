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
from project.camera_readers.realsense import RealSenseCamera
import mediapy
from pathlib import Path
from project.utils.paths import repo_root

# %%
cam = RealSenseCamera()

# %%
view1 = cam.get_view()

# %%
view2 = cam.get_view()

# %%
mediapy.show_images([view1.color, view2.color])

# %%
base_path = Path("../data/view_pairs/4")
view1.save(base_path / 'view1')
view2.save(base_path / 'view2')
