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
# # Monumental plantage shed take-home with BA
#
# Here I'll solve the monumental vision take-home with bundle adjustment. 
#
# Our optimization variables are the camera poses and the april tag (landmark) locations.
#
# The april tags are our landmarks, and the camera pose
#
# 1. Load the video and the camera model
# 2. Detect the april tags.
# 3. Set up the BA

# %%
# %load_ext autoreload
# %autoreload 2
import json
from project.utils.camera import Intrinsics, Camera
from project.utils.spatial import Pose
from pathlib import Path
import project.utils.rerun_utils as rr_utils
import rerun as rr
import dt_apriltags as april
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import imageio.v3 as iio
from project.utils.image import to_greyscale
import mediapy
from project.utils.paths import repo_root

# %%
TAG_SIDE_LENGTH = 42

# %%
base_path = repo_root() / "data/monumental_take_home"

# %%
camera_json = json.loads((base_path / 'cam.json').read_text())
camera_json

# %%
intrinsics = Intrinsics.from_pinhole_params(
    camera_json['width'],
    camera_json['height'],
    camera_json['fx'],
    camera_json['fy'],
    camera_json['px'],
    camera_json['py'],
    np.array(camera_json['dist_coeffs'])
)
intrinsics

# %%
# if 'vid' not in globals() or 'greyscale_video' not in globals():
vid = iio.imiter(base_path / 'plantage_shed.mp4')

# %%
if 'detector' not in globals():
    detector = april.Detector(families="tagStandard52h13", nthreads=16)


# %%
@dataclass
class DetectedTag:
    tag_id: int
    corners: np.ndarray

@dataclass
class ImageDetectedTags:
    detected_tags: list[DetectedTag]

    def unique_ids(self) -> set[int]:
        return set(dt.tag_id for dt in self.detected_tags)

    def __len__(self) -> int:
        return len(self.detected_tags)

    def __getitem__(self, idx: int) -> DetectedTag:
        return self.detected_tags[idx]

@dataclass
class VideoTagDetections:
    detections: list[ImageDetectedTags]

    def unique_ids(self) -> set[int]:
        return set.union(*[dt.unique_ids() for dt in self.detections])

    def __len__(self):
        return len(self.detections)

    def __getitem__(self, idx: int) -> ImageDetectedTags:
        return self.detections[idx]


# %%
real_vid_frame_indices = []
tag_detections_lis = []
for i, frame in tqdm(enumerate(vid)):

    if (i % 25) != 0:
        continue

    real_vid_frame_indices.append(i)

    grey = to_greyscale(frame)
    april_res = detector.detect(grey)
    detected_tags = []
    for tag_april in april_res:
        tag_id = tag_april.tag_id
        corners = np.array(tag_april.corners)
        detected_tags.append(DetectedTag(
            tag_id, corners
        ))
    tag_detections_lis.append(ImageDetectedTags(detected_tags))

    # if len(tag_detections_lis) == 3:
    #     break

tag_detections = VideoTagDetections(tag_detections_lis)

# %%
unique_ids = tag_detections.unique_ids()
len(unique_ids)

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

initial_values = Values()
optimized_keys = []

for frame_idx in range(len(tag_detections)):
    cam_pose_key = f"cam_pose{frame_idx}"

    if frame_idx != 0:
        optimized_keys.append(cam_pose_key)

    initial_values[cam_pose_key] = sf.Pose3.identity()
    
for tag_id in tag_detections.unique_ids():
    tag_pose_key = f"tag_pose{tag_id}"
    optimized_keys.append(tag_pose_key)
    initial_values[tag_pose_key] = pose_to_sf(Pose.identity().with_translation(np.array([0, 0, 100])))

initial_values['camera_cal'] = sf.LinearCameraCal(
    [float(intrinsics.fx), float(intrinsics.fy)],
    [float(intrinsics.cx), float(intrinsics.cy)]
)

initial_values['epsilon'] = sf.numeric_epsilon


for i, (x_m, y_m) in enumerate([[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]]):
    initial_values[f'dir{i}'] = sf.V2(x_m, y_m)


# %%
def factor(camera_pose: sf.Pose3, tag_pose: sf.Pose3, camera_cal: sf.LinearCameraCal, dir: sf.V2, corner_in_img: sf.V2, epsilon: sf.Scalar):
    tag_R_matrix = tag_pose.R.to_rotation_matrix()
    tag_pose_x_dir = tag_R_matrix[:3, 0]
    tag_pose_y_dir = tag_R_matrix[:3, 1]

    tag_t = tag_pose.t

    corner_in_world = tag_t + (tag_pose_x_dir * dir[0] * TAG_SIDE_LENGTH / 2) + (tag_pose_y_dir * dir[1] * TAG_SIDE_LENGTH / 2)
    corner_in_cam = camera_pose * corner_in_world

    projected, _ = camera_cal.pixel_from_camera_point(corner_in_cam, epsilon)

    return projected - corner_in_img

camera_poses = []
factors = []

for frame_idx in tqdm(range(len(tag_detections))):

    frame_detection = tag_detections[frame_idx]

    for tag_detection in frame_detection.detected_tags:
        tag_id = tag_detection.tag_id

        undistorted_corners = intrinsics.undistort_pixels(tag_detection.corners)
        
        undistorted_corners_sf = [
            sf.V2(*c.tolist())
            for c in undistorted_corners
        ]

        for i in range(4):

            corner_key = f'corner{frame_idx}_{tag_id}_{i}'
            initial_values[corner_key] = undistorted_corners[i]

            factors.append(Factor(
                residual=factor,
                keys=[f'cam_pose{frame_idx}', f'tag_pose{tag_id}', 'camera_cal', f"dir{i}", corner_key, 'epsilon']
            ))


print(len(factors))

# %%
params = Optimizer.Params(
    debug_stats=True
)

optimizer = Optimizer(
    factors=factors,
    optimized_keys=optimized_keys,
    params=params
)

# %%
res = optimizer.optimize(initial_values)

# %%
res._stats.iterations[-1]

# %%
res

# %%
rr_utils.init("plantage_shed")
rr.log('/', rr.Clear(recursive=True))

# %%
opt_values = res.optimized_values

# %%
camera_poses = []
for frame_idx in range(len(tag_detections)):
    cam_pose = sf_to_pose(opt_values[f'cam_pose{frame_idx}']).inv
    camera_poses.append(cam_pose)

    camera = Camera(
        intrinsics,
        cam_pose
    )
    rr_utils.log_camera(f"/cams/cam{frame_idx}", camera)

# %%
tag_poses = {
    i: sf_to_pose(opt_values[f"tag_pose{i}"])
    for i in tag_detections.unique_ids()
}

# %%
for i, pose in enumerate(tag_poses.values()):
    rr.log(f"/tags/tag{i}", rr_utils.transform(pose))
    rr.log(f"/tags/tag{i}/triad", rr_utils.triad())

# %%
rr.log("/camera_trajectory/path", rr.LineStrips3D([p.tvec for p in camera_poses]))
rr.log("/camera_trajectory/start", rr.Points3D(camera_poses[0].tvec, radii=5.0))

# %%
real_vid_frames_set = set(real_vid_frame_indices)

# %%
corners_in_tag = np.array([[-1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]]) * TAG_SIDE_LENGTH / 2

# %%
from project.utils.colors import Colors
from project.utils.drawing import draw_lines


vid = iio.imiter(base_path / 'plantage_shed.mp4')
overlaid_vid = []

keyframe_idx = 0
for i, frame in tqdm(enumerate(vid)):
    if i not in real_vid_frames_set:
        # overlaid_vid.append(frame)
        continue
    

    cam_pose = camera_poses[keyframe_idx]
    keyframe_idx += 1

    for tag_id, tag_pose in tag_poses.items():
        corners_in_world = tag_pose.apply(corners_in_tag)

        corners_in_cam = cam_pose.inv.apply(corners_in_world)

        corners_pix = intrinsics.project_points(corners_in_cam.astype(float))

        if np.any(
            (corners_pix[:, 0] >= intrinsics.width) 
            | (corners_pix[:, 0] < 0) 
            | (corners_pix[:, 1] >= intrinsics.height) 
            | (corners_pix[:, 1] < 0)
        ):
            continue

        frame = draw_lines(frame, corners_pix, np.roll(corners_pix, 1, axis=0), color=Colors.GREEN)

    overlaid_vid.append(frame)
    

# %%
mediapy.show_video(overlaid_vid, fps=2)
