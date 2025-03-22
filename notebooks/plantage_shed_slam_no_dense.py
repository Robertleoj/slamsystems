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
# # Plantage shed bundle adjustment from scratch
#
# Here I'll do bundle adjustment completely from scratch, on the plantage shed video. 

# %%
# %load_ext autoreload
# %autoreload 2
import json
from project.utils.camera import Intrinsics, Camera
from project.utils.paths import repo_root
import project.utils.rerun_utils as rr_utils
import rerun as rr
import numpy as np
from tqdm import tqdm
import imageio.v3 as iio
from project.utils.paths import repo_root
from project.tag_slam.default import TagSlam

# %%
TAG_SIDE_LENGTH = 42

# %%
base_path = repo_root() / "data/monumental_take_home"
camera_json = json.loads((base_path / 'cam.json').read_text())
intrinsics = Intrinsics.from_pinhole_params(
    camera_json['width'],
    camera_json['height'],
    camera_json['fx'],
    camera_json['fy'],
    camera_json['px'],
    camera_json['py'],
    np.array(camera_json['dist_coeffs'])
)

# %%
rr_utils.init("tagslam_with_ba")

# %%
vid = iio.imiter(base_path / 'plantage_shed.mp4')

# %%
tagslam = TagSlam(TAG_SIDE_LENGTH, intrinsics)


# %%
def log_state(path: str, tagslam: TagSlam):

    rr.log(f"{path}/camera_trajectory/path", rr.LineStrips3D([p.tvec for p in tagslam.camera_poses]))

    rr.log(f"{path}/camera_trajectory/start", rr.Points3D(tagslam.camera_poses[0].tvec, radii=5.0))

    for tag_id, tag in tagslam.landmarks.items():
        rr.log(f"{path}/tags/tag{tag_id}", rr_utils.transform(tag.pose))
        rr.log(f"{path}/tags/tag{tag_id}/triad", rr_utils.triad())

    for keyframe_idx in tagslam.keyframe_indices:
        cam_pose = tagslam.camera_poses[keyframe_idx]

        camera = Camera(
            tagslam.cam_intrinsics,
            cam_pose
        )

        rr_utils.log_camera(f"{path}/cams/cam{keyframe_idx}", camera)



# %%
depth_map_frames = []
for i, frame in tqdm(enumerate(vid)):
    tagslam.process_frame(frame)

    if i % 100 == 0:
        rr.set_time_sequence("slamming", i)
        log_state("/live_slam", tagslam)

# %%
tagslam.refine()

# %%
log_state("/refined", tagslam)

# %%
reconstructions = tagslam.get_dense_reconstruction_vo()

# %%
for keyframe_idx, (points, colors) in reconstructions.items():
    rr.log(f"/dense/frame_{keyframe_idx}", rr.Points3D(points, colors=colors))
