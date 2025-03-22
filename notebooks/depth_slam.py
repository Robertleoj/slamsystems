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
# # Depth camera slam
#
# Let's try to do slam with the konowledge I have now, using a depth camera!
#
# I need to build the frontend, and the backend.

# %% [markdown]
# ## Frontend
#
# We'll use optical flow for the frontend.

# %% [markdown]
# ## Backend
# For the backend, I'll use symforce to do the bundle adjustment. 
#
# I'll first just do full bundle adjustment on the poses and landmarks, and then try going for a pose graph for efficiency.
#
# ### Loop closure
#
# I'll try to develop some basic loop closure.

# %% [markdown]
# ## Dense reconstruction
# Since we have a depth camera, we can do dense reconstruction using point cloud fusing. 

# %%
# %load_ext autoreload
# %autoreload 2
from project.depth_slam import DepthSlam
from project.utils.camera import Video
from project.utils.paths import repo_root
import logging
import project.utils.rerun_utils as rr_utils
import numpy as np
import rerun as rr
import mediapy
from tqdm import tqdm

# %%
logging.basicConfig(level=logging.INFO)

# %%
vid_path = repo_root() / 'data/videos/vid2'

vid = Video.load(vid_path)

# %%
from project.utils.camera.camera_params import Camera
from project.utils.colors import Colors


def log_state(path: str, slam: DepthSlam):

    rr.log(f"{path}/camera_trajectory/start", rr.Points3D(slam.frames[0].camera_pose.tvec, radii=5.0), static=True)

    rr.log(f"{path}/camera_trajectory/path", rr.LineStrips3D([f.camera_pose.tvec for f in slam.frames]))


    curr_landmarks = set(slam.frames[-1].landmark_ids)
    landmark_locations = []
    landmark_colors = []

    for lid, landmark in slam.map.landmarks.items():
        landmark_locations.append(landmark.loc)
        if lid in curr_landmarks:
            landmark_colors.append(Colors.GREEN)
        else:
            landmark_colors.append(Colors.RED)
    
    rr.log(f"{path}/landmarks", rr.Points3D(
        np.array(landmark_locations)
    , colors=landmark_colors))

    # for i in range(0, len(slam.camera_poses), 10):
    for keyframe_idx in slam.keyframe_indices:
        cam_pose = slam.frames[keyframe_idx].camera_pose

        camera = Camera(
            slam.intrinsics,
            cam_pose
        )

        rr_utils.log_camera(f"{path}/cams/cam{keyframe_idx}", camera)

        pc = slam.keyframe_pointclouds_in_cam[keyframe_idx]
        points = cam_pose.apply(pc.points)
        rr.log(f"{path}/pointclouds/pointloud{keyframe_idx}", rr.Points3D(points, colors=pc.colors))


# %%
depth_slam = DepthSlam(vid.intrinsics)

# %%
rr_utils.init("depth_slam")

# %%
i = 0

# %%
# view = vid[i]
# rr.set_time_sequence("slam", i)
# depth_slam.process_frame(view.color, view.depth_assumed)
# log_state("/slam", depth_slam)

# i = i + 1

# %%
for i, view in tqdm(enumerate(vid)):
    rr.set_time_sequence("slam", i)
    depth_slam.process_frame(view.color, view.depth_assumed)
    log_state("/slam", depth_slam)


# %%
mediapy.show_video(vid.color)

# %%
# import open3d as o3d

# full_pointcloud = o3d.geometry.PointCloud()

# for keyframe_index in tqdm(depth_slam.keyframe_indices):

#     kf0_pose = depth_slam.frames[keyframe_index].camera_pose
#     kf0_view = depth_slam.keyframe_views[keyframe_index]

#     y, x = np.where((kf0_view.depth_assumed > 0) & (kf0_view.depth_assumed < 500))

#     valid_depths = kf0_view.depth_assumed[y, x]
#     valid_colors = kf0_view.color[y, x] / 255
#     pixels = np.array([x, y]).T

#     points3d_cam = depth_slam.intrinsics.unproject_depths(pixels, valid_depths)
#     points3d_world = kf0_pose.apply(points3d_cam)

#     # rr.log("/pointclouds/test_kf0", rr.Points3D(points3d_world, colors=valid_colors))

#     cloud = o3d.geometry.PointCloud()
#     cloud.points = o3d.utility.Vector3dVector(points3d_world)
#     cloud.colors = o3d.utility.Vector3dVector(valid_colors)

#     full_pointcloud += cloud

# rr.log("/pointclouds/full_cloud", rr.Points3D(np.asarray(full_pointcloud.points), colors=np.asarray(full_pointcloud.colors)), static=True)

# voxel_size = 2.0

# cloud_small, trace_indices, _ = full_pointcloud.voxel_down_sample_and_trace(voxel_size, full_pointcloud.get_min_bound(), full_pointcloud.get_max_bound())

# colors = np.asarray(full_pointcloud.colors)

# print(trace_indices.shape)
# print(np.asarray(cloud_small.points).shape)

# cloud_small.colors = o3d.utility.Vector3dVector(
#     colors[np.max(trace_indices, axis=1)]
# )

# rr.log("/pointclouds/full_cloud_small", rr.Points3D(np.asarray(cloud_small.points), colors=np.asarray(cloud_small.colors)), static=True)

# cloud_small.estimate_normals()

# radii = np.array([0.5, 1.0, 2.0, 4.0]) * 10

# # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud_small, depth=10)
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud_small, radii=o3d.utility.DoubleVector(radii))

# rr.log("/meshes/reconstruction_mesh", rr.Mesh3D(
#     vertex_positions=np.asarray(mesh.vertices),
#     triangle_indices=np.asarray(mesh.triangles),
#     vertex_colors=np.asarray(mesh.vertex_colors)
# ), static=True)


# %%
