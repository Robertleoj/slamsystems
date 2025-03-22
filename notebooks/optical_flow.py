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
from project.utils.image import to_greyscale
import project.utils.rerun_utils as rr_utils
import rerun as rr
import rerun.blueprint as rrb
import numpy as np
import cv2
from project.utils.drawing import draw_pixels, draw_lines
from project.utils.features import get_good_features_to_track
from project.utils.optical_flow import lk_optical_flow
from project.utils.colors import Colors
import mediapy
from project.utils.paths import repo_root

# %%

# %% [markdown]
# # Take a couple of imgs

# %%
cam_reader = RealSenseCamera()

# %%
# view1 = cam_reader.get_view()

# %%
# view2 = cam_reader.get_view()

# %%
base_path = repo_root() / "data" / "view_pairs" / "3"

# %%
# view1.save(base_path / "view1")
# view2.save(base_path / "view2")

# %%
view1 = View.load(base_path / "view1")
view2 = View.load(base_path / "view2")

# %%
grey1 = to_greyscale(view1.color)
grey2 = to_greyscale(view2.color)

# %%
mediapy.show_images([view1.color, view2.color, grey1, grey2], columns=2)

# %%
features = cv2.goodFeaturesToTrack(grey1, 100, 0.3, 7)
display(features.shape)
features.squeeze()

# %%
mediapy.show_image(draw_pixels(view1.color, features.squeeze(), color=Colors.RED, radius=4))

# %%
pts2, status, err = cv2.calcOpticalFlowPyrLK(view1.color, view2.color, features, None)

# %%
display(pts2.shape)
display(pts2.squeeze())
display(status.squeeze())
display(err.squeeze())

# %%
debug_img = view2.color.copy()
debug_img = draw_pixels(debug_img, features.squeeze(), Colors.RED, radius=3)
debug_img = draw_pixels(debug_img, pts2.squeeze(), Colors.GREEN, radius=3)
debug_img = draw_lines(debug_img, features.squeeze(), pts2.squeeze(), Colors.GOLDEN_YELLOW)
mediapy.show_image(debug_img)

# %%
rr_utils.init()

# %%
curr_img = cam_reader.get_view().color
curr_pts = get_good_features_to_track(curr_img)


vid = [curr_img]
vid_points = [curr_pts]
redetect_indices = [0]

min_features_before_redetect = 15

color = Colors.RED

for i in range(1, 100):
    next_img = cam_reader.get_view().color

    vid.append(next_img)

    # try matching against the features we already have
    if len(curr_pts) > 1:
        matched_points, success_mask, err = lk_optical_flow(curr_img, next_img, curr_pts)

        curr_pts = matched_points[success_mask]



    # first check if we have enough features to check
    # if not, re-get features
    if len(curr_pts) < min_features_before_redetect:
        curr_pts = get_good_features_to_track(next_img)

        if color == Colors.RED:
            color = Colors.GREEN
        else:
            color = Colors.RED

        redetect_indices.append(i)


    curr_img = next_img
    vid_points.append(curr_pts)
    
    drawn = curr_img.copy()
    if len(curr_pts) > 0:
        drawn = draw_pixels(drawn,curr_pts, color)


    cv2.imshow("vid", drawn)
    cv2.waitKey(1)

# %%
accumulated_points = []

for i, (img, pts) in enumerate(zip(vid, vid_points)):

    if i in redetect_indices:
        accumulated_points.clear()

    accumulated_points.append(pts)

    rr.set_time_sequence("vid", i)

    rr.log("/vid/img", rr.Image(img))
    for i, p_set in enumerate(accumulated_points):
        rr.log(f"/vid/pts/{i}", rr.Points2D(p_set, radii=2.0))

    # rr.log("/vid/pts/lines", rr.LineStrips2D(
    #     np.array(accumulated_points),
    #     radii=2.0
    # ))


    
    

# %%
