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
import symforce
if 'eps_set' not in globals():
    symforce.set_epsilon_to_symbol()
    eps_set = True


import json
import einops
from project.utils.camera import Intrinsics, Camera
from project.utils.markers import DetectedTag
from project.utils.spatial import Pose
from project.utils.markers import get_corners_in_tag
from pathlib import Path
import project.utils.rerun_utils as rr_utils
import rerun as rr
import dt_apriltags as april
from typing import cast
from project.utils.symforce_utils import pose_to_sf
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import imageio.v3 as iio
from project.utils.image import to_greyscale
import random
import mediapy
from project.utils.paths import repo_root
from project.tag_slam import TagSlam

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
vid = iio.imiter(base_path / 'plantage_shed.mp4')

# %%
tagslam = TagSlam(TAG_SIDE_LENGTH, intrinsics)

# %%
for i, frame in tqdm(enumerate(vid)):
    if i > 20:
        break
    tagslam.process_frame(frame)

# %%
rr_utils.init("tagslam")

# %%

for i, cam_pose in enumerate(tagslam.camera_poses):
    camera = Camera(
        intrinsics,
        cam_pose
    )
    rr_utils.log_camera(f"/tagslam/cams/cam{i}", camera)


# %%
rr.log("/camera_trajectory/path", rr.LineStrips3D([p.tvec for p in tagslam.camera_poses]))
rr.log("/camera_trajectory/start", rr.Points3D(tagslam.camera_poses[0].tvec, radii=5.0))
for i, (tag_id, tag3d) in enumerate(    tagslam.landmarks.items()):
    rr.log(f"/tags/tag{i}", rr_utils.transform(tag3d.pose))
    rr.log(f"/tags/tag{i}/triad", rr_utils.triad())

# %%
import symforce.symbolic as sf
from symforce import codegen
from symforce.notebook_util import display_code_file
from symforce.notebook_util import set_notebook_defaults
from symforce.values import Values
from IPython.core.display import HTML
from IPython.display import display


set_notebook_defaults()

display(HTML("""
<style>
div.highlight {
    background: #1e1e1e !important;  /* Dark background */
    color: #dcdcdc !important;  /* Light text */
}
</style>
"""))


# %%
class TagSlamJacobian:

    def __init__(self, intrinsics: Intrinsics, tag_side_length: float):
        
        camera = sf.LinearCameraCal(
            [intrinsics.fx, intrinsics.fy],
            [intrinsics.cx, intrinsics.cy]
        )

        self.tag_pose = sf.Pose3.symbolic('T')
        self.cam_pose = sf.Pose3.symbolic('C')
        self.epsilon = sf.epsilon()

        self.measurement = sf.V8.symbolic("z")

        observation = sf.V8()
        for i, tag_corner in enumerate(get_corners_in_tag(tag_side_length)):
            corner_sf = sf.V3(tag_corner)
            corner_in_world = cast(sf.V3, self.tag_pose * corner_sf)
            corner_in_cam = cast(sf.V3, self.cam_pose.inverse() * corner_in_world)

            tag_pixels, _ = camera.pixel_from_camera_point(corner_in_cam, self.epsilon)

            observation[2 * i : 2* (i + 1)] = tag_pixels

        self.error = self.measurement - observation
        self.de_dcam = self.error.jacobian(self.cam_pose)
        self.de_dtag = self.error.jacobian(self.tag_pose)


    def __call__(self, cam_pose: Pose, tag_pose: Pose, measurement: np.ndarray):
        """Compute errors and jacobians


        Returns:
            residual: shape 8 x 1 residual
            de_dcam: 8 x 6 jacobian of the error w.r.t. the camera pose
            de_dtag: 8 x 6 jacobian of the error w.r.t. the tag pose
        """
        sub_dict = {
            self.cam_pose: pose_to_sf(cam_pose),
            self.tag_pose: pose_to_sf(tag_pose),
            self.measurement: sf.V8(measurement),
            self.epsilon: sf.numeric_epsilon
        }

        return (
            self.error.subs(sub_dict).to_numpy(),
            self.de_dcam.subs(sub_dict).to_numpy(), 
            self.de_dtag.subs(sub_dict).to_numpy(),
        )

jacobian = TagSlamJacobian(intrinsics, TAG_SIDE_LENGTH)

res, de_dcam, de_dtag = jacobian.__call__(Pose.identity(), Pose.identity(), np.zeros(8))
print(res.shape, de_dcam.shape, de_dtag.shape)

# %%
# CodeGen if you want that

# def codegen_functions(intrinsics: Intrinsics, tag_side_length: float):
    
#     camera = sf.LinearCameraCal(
#         [intrinsics.fx, intrinsics.fy],
#         [intrinsics.cx, intrinsics.cy]
#     )

#     corners_in_tag = [
#         sf.V3(c)
#         for c in get_corners_in_tag(tag_side_length)
#     ]

#     def compute_error(tag_pose: sf.Pose3, cam_pose: sf.Pose3, measurement: sf.V8, epsilon: sf.Scalar=0):

#         observation = sf.V8()
#         for i, corner_sf in enumerate(corners_in_tag):
#             corner_in_world = cast(sf.V3, tag_pose * corner_sf)
#             corner_in_cam = cast(sf.V3, cam_pose * corner_in_world)

#             tag_pixels, _ = camera.pixel_from_camera_point(corner_in_cam, epsilon)

#             observation[2 * i : 2* (i + 1)] = tag_pixels

#         error = measurement - observation

#         return error

#     return compute_error



# compute_err = codegen_functions(intrinsics, TAG_SIDE_LENGTH)

# compute_err_codegen = codegen.Codegen.function(
#     func=compute_err,
#     config=codegen.PythonConfig()
# )

# func = compute_err_codegen.generate_function()

# display_code_file(func.generated_files[0], "python")

# %% [markdown]
# ## Toy problem

# %%
keyframes = range(0, len(tagslam.camera_poses), 3)

cam_poses: list[Pose] = []
measurements: list[dict[int, np.ndarray]] = []

tag_ids = np.random.choice(list(tagslam.landmarks.keys()), size=20, replace=False)

found_tags = set()

for frame_idx in keyframes:
    cam_poses.append(Pose.random(0.1) @ tagslam.camera_poses[frame_idx])

    frame_measurements = {}
    for tag_id in tag_ids:
        detection = tagslam.detections[frame_idx].get(tag_id, None)
        if detection is None:
            continue

        frame_measurements[tag_id] = detection.corners.reshape(8, 1)
        found_tags.add(tag_id)

    measurements.append(frame_measurements)

not_found = set(tag_ids) - found_tags

tag_ids = list(found_tags)

tag_poses = {
    tag_id: Pose.random(0.1) @ tagslam.landmarks[tag_id].pose
    for tag_id in tag_ids
}

tag_id_to_idx = {
    tag_id: i
    for i, tag_id in enumerate(tag_ids)
}

print(f"num found tags: {len(tag_ids)}")


# %%
def norm_clip(x, max_norm):
    norm = np.linalg.norm(x)
    if norm > max_norm:
        x = x * (max_norm / norm)  # Scale down to max_norm
    return x


# %%
def get_H_g(cam_poses: list[Pose], frame_measurements: list[dict[int, np.ndarray]], tag_poses: dict[int, Pose]) -> tuple[np.ndarray, np.ndarray]:
    m = len(cam_poses)
    n = len(tag_poses)

    x_dim = 6 * (m + n)

    H = np.zeros((x_dim, x_dim))
    g = np.zeros(x_dim)

    total_error = 0

    for i, (cam_pose, measurements) in enumerate(zip(cam_poses, frame_measurements)):
        for tag_id, tag_pose in tag_poses.items():
            if tag_id not in measurements:
                continue

            measurement = measurements[tag_id]

            j = tag_id_to_idx[tag_id]

            err, de_dcam, de_dtag = jacobian(cam_pose, tag_pose, measurement)


            i_block = slice(6* i, 6* (i + 1))
            j_block = slice((6 * m) + 6 * j, (6 * m) + 6 * (j + 1))

            H[i_block, i_block] += einops.einsum(de_dcam, de_dcam, "r d1, r d2 -> d1 d2")
            H[j_block, j_block] += einops.einsum(de_dtag, de_dtag, 'r d1, r d2 -> d1 d2')

            H_ij = einops.einsum(de_dcam, de_dtag, "r dc, r dt -> dc dt")
            H[i_block, j_block] += H_ij
            H[j_block, i_block] += H_ij.T

            total_error += (err ** 2).sum()
            g[i_block] -= einops.einsum(de_dcam, err, "r d, r o -> d")
            g[j_block] -= einops.einsum(de_dtag, err, "r d, r o -> d")

    rr.log("/tensors/err", rr.Scalar(total_error))
    rr.log("/tensors/H", rr.Tensor(np.clip(H, -1, 1)), static=True)
    rr.log("/tensors/g", rr.Tensor(g), static=True)

    return H, g



H, g = get_H_g(cam_poses, measurements, tag_poses)
display(H.shape, g.shape)



# %%
from project.utils.symforce_utils import sf_to_pose


def gauss_newton_step(cam_poses: list[Pose], measurements: list[dict[int, np.ndarray]], tag_poses: dict[int, Pose]) -> tuple[list[Pose], dict[int, Pose]]:
    H, g = get_H_g(cam_poses, measurements, tag_poses)
    
    delta_x = np.linalg.solve(H, g)
    # clip the norm
    delta_x = norm_clip(delta_x, 500)
    

    new_poses = []
    new_tag_poses = {}

    delta_x_cam = delta_x[:6 * len(cam_poses)]
    delta_x_tag = delta_x[6 * len(cam_poses):]

    for i, cam_pose in enumerate(cam_poses):
        pose_update_vec = delta_x_cam[i * 6: (i + 1) * 6]
        pose_update = sf_to_pose(sf.Pose3.from_tangent(pose_update_vec.flatten().tolist(), sf.numeric_epsilon))

        new_pose = pose_update @ cam_pose
        new_poses.append(new_pose)

    for tag_id, tag_pose in tag_poses.items():
        j = tag_id_to_idx[tag_id]

        pose_update_vec = delta_x_tag[j * 6: (j + 1) * 6]
        pose_update = sf_to_pose(sf.Pose3.from_tangent(pose_update_vec.flatten().tolist(), sf.numeric_epsilon))

        new_pose = pose_update @ tag_pose

        new_tag_poses[tag_id] = new_pose

    return new_poses, new_tag_poses


# %%
def log_state(path: str, cam_poses: list[Pose], tag_poses: dict[int, Pose]):
    rr.log(f"{path}/camera_trajectory/path", rr.LineStrips3D([p.tvec for p in cam_poses]))
    rr.log(f"{path}/camera_trajectory/start", rr.Points3D(cam_poses[0].tvec, radii=5.0))
    for tag_id, tag_pose in tag_poses.items():
        rr.log(f"{path}/tags/tag{tag_id}", rr_utils.transform(tag_pose))
        rr.log(f"{path}/tags/tag{tag_id}/triad", rr_utils.triad())

    for i, cam_pose in enumerate(cam_poses):
        camera = Camera(
            intrinsics,
            cam_pose
        )
        rr_utils.log_camera(f"{path}/cams/cam{i}", camera)


# %%

timeline_name = "gauss_newton2"
viewer_root = "/gauss_newton"
steps_cam_poses = [cam_poses]
steps_tag_poses = [tag_poses]

time_seq = 0

rr.set_time_seconds(timeline_name, time_seq)
log_state(viewer_root, cam_poses, tag_poses)


# %%

for i in range(50):
    time_seq += 1
    rr.set_time_seconds(timeline_name, time_seq)
    new_cam, new_tags = gauss_newton_step(steps_cam_poses[-1], measurements, steps_tag_poses[-1])

    log_state(viewer_root, new_cam, new_tags)
    steps_cam_poses.append(new_cam)
    steps_tag_poses.append(new_tags)


# %% [markdown]
# # Not fucking working bro
#
