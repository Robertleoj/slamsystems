# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: project-ePUsKrUH-py3.10
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
from project.camera_readers.realsense import RealSenseCamera
import mediapy
from project.utils.spatial.pose import rz_R, Pose
from project.utils.features import orb_detect, orb_detect_and_compute, draw_keypoints, match_feature_descriptors, draw_matches, get_flann_matcher_for_orb, get_good_matches
import numpy as np
from scipy.linalg import null_space
from itertools import combinations
import viser
import cv2

# %% [markdown]
# # Try camera out

# %%
if 'cam' not in globals():
    cam = RealSenseCamera()

# %%
view = cam.get_view()

# %%
print(view.color.dtype)
print(view.depth.dtype)


# %%
view.depth.mean()

# %%
mediapy.show_image(view.color)

# %% [markdown]
# # Detect features

# %%
orb_detector = cv2.ORB.create()

# %%
keypoints, descriptors = orb_detect_and_compute(view.color, orb=orb_detector)

# %%
keypoints[0]

# %%
descriptors[0]

# %%
len(keypoints), descriptors.shape

# %%
drawn = draw_keypoints(view.color, keypoints)

# %%
mediapy.show_image(drawn)

# %%
orb = cv2.ORB.create(nfeatures=100)

# %%
cv2.namedWindow("imgs", cv2.WINDOW_NORMAL)
while True:
    view = cam.get_view()
    keypoints = orb_detect(view.color, orb=orb)
    drawn = draw_keypoints(view.color, keypoints=keypoints)
    cv2.imshow('imgs', drawn)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
cv2.destroyAllWindows()
    

# %%
view_1 = cam.get_view()

# %%
view_2 = cam.get_view()

# %%
mediapy.show_images([view_1.color, view_2.color])

# %%
keypoints_1, descriptors_1 = orb_detect_and_compute(view_1.color, orb=orb)
keypoints_2, descriptors_2 = orb_detect_and_compute(view_2.color, orb=orb)

# %%
kp1_drawn = draw_keypoints(view_1.color, keypoints=keypoints_1)
kp2_drawn = draw_keypoints(view_2.color, keypoints=keypoints_2)

# %%
mediapy.show_images([kp1_drawn, kp2_drawn])

# %% [markdown]
# # Try matching features

# %%
# Create Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = match_feature_descriptors(descriptors_1, descriptors_2, bf)

# %%
matches = sorted(matches, key=lambda x: x.distance)

# %%
drawn_matches = draw_matches(view_1.color, keypoints_1, view_2.color, keypoints_2, matches[:10])
mediapy.show_image(drawn_matches)

# %%
flann = get_flann_matcher_for_orb()

# %%
matches = match_feature_descriptors(descriptors_1, descriptors_2, flann)

# %%
good_matches = get_good_matches(matches)

# %%
drawn_matches = draw_matches(view_1.color, keypoints_1, view_2.color, keypoints_2, matches[:20])
mediapy.show_image(drawn_matches)

# %% [markdown]
# # Estimate camera motion

# %%
points_1_lis = []
points_2_lis = []
for match in good_matches:
    point_1_idx = match.query_idx
    point_1 = keypoints_1[point_1_idx].point
    points_1_lis.append(point_1.to_arr())

    point_2_idx = match.target_idx
    point_2 = keypoints_2[point_2_idx].point
    points_2_lis.append(point_2.to_arr())

points_1 = np.array(points_1_lis)
points_2 = np.array(points_2_lis)

# %%
points_1.shape, points_2.shape

# %%
camera_matrix = view_1.intrinsics.camera_matrix
dist_coeffs = view_1.intrinsics.distortion_parameters

# %%
points_1_normalized = cv2.undistortPoints(points_1, camera_matrix, dist_coeffs, None)
points_2_normalized = cv2.undistortPoints(points_2, camera_matrix, dist_coeffs, None)

# %%
points_1_normalized[:2]


# %%
def try_solve(points_1: np.ndarray, points_2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    assert points_1.shape == (8, 2)
    assert points_2.shape == (8, 2)

    constraint_matrix = np.zeros((8, 9), dtype=np.float32)

    # pt = (u, v)
    constraint_matrix[:, 0] = points_2[:, 0] * points_1[:, 0] # u2 * u1
    constraint_matrix[:, 1] = points_2[:, 0] * points_1[:, 1] # u2 * v1
    constraint_matrix[:, 2] = points_2[:, 0] # u2
    constraint_matrix[:, 3] = points_2[:, 1] * points_1[:, 0] # v2 * u1
    constraint_matrix[:, 4] = points_2[:, 1] * points_1[:, 1] # v2 * v1
    constraint_matrix[:, 5] = points_2[:, 1] # v2
    constraint_matrix[:, 6] = points_1[:, 0] # u1
    constraint_matrix[:, 7] = points_1[:, 1] # v1
    constraint_matrix[:, 8] = 1

    constraint_matrix_rank = np.linalg.matrix_rank(constraint_matrix)

    if constraint_matrix_rank < 8:
        print(f"Matrix has rank {constraint_matrix_rank}")
        return None

    
    nspace_basis = null_space(constraint_matrix).squeeze()
    assert nspace_basis.shape == (9,)

    # normalize to norm 1
    nspace_basis /= np.linalg.norm(nspace_basis)
    
    E_init = nspace_basis.reshape(3, 3)
    print(E_init)

    # now we put E to the correct form

    U, S, Vh = np.linalg.svd(E_init)
    print(S)
    print(f"{U.shape=}")
    print(f"{S.shape=}")
    print(f"{Vh.shape=}")

    new_S = np.diag([1.0, 1.0, 0])

    return U, new_S, Vh

# %%
num_correspondances = len(points_1_normalized)
num_correspondances

# %%
import random
def random_comb(n, k):
    return random.sample(list(range(n)), k)


# %%
idx_sample = random_comb(num_correspondances, 8)
print(idx_sample)
points_1_sample = points_1_normalized[idx_sample].squeeze()
points_2_sample = points_2_normalized[idx_sample].squeeze()
display(points_1_sample.shape, points_2_sample.shape)
out = try_solve(points_1_sample, points_2_sample)

# %%
assert out is not None
U, S, Vh = out

# %%
t_sk_1 = U @ rz_R(np.pi / 2) @ S @ U.T
R_1 = U @ rz_R(np.pi / 2).T @ Vh


if np.linalg.det(R_1) < 0:
    R_1 = -R_1

R_1_det = np.linalg.det(R_1)
print(f"{R_1_det=}")

t_sk_2 = U @ rz_R(- np.pi / 2) @ S @ U.T
R_2 = U @ rz_R(-np.pi / 2).T @ Vh

if np.linalg.det(R_2) < 0:
    R_2 = - R_2

R_2_det = np.linalg.det(R_2)
print(f"{R_2_det=}")



def sk_to_vec(sk: np.ndarray) -> np.ndarray:
    return np.array([
        -sk[1, 2],
        sk[0, 2],
        -sk[0, 1]
    ])

t_1 = sk_to_vec(t_sk_1)
t_2 = sk_to_vec(t_sk_2)

possible_solutions = [
    (t_1, R_1),
    (-t_1, R_1),
    (t_2, R_2),
    (-t_2, R_2)
]


# %%
def triangulate_point(ray1_origin, ray1_direction, ray2_origin, ray2_direction):
    ray1_direction = ray1_direction / np.linalg.norm(ray1_direction)
    ray2_direction = ray2_direction / np.linalg.norm(ray2_direction)

    cross_dir = np.cross(ray1_direction, ray2_direction)
    denom = np.linalg.norm(cross_dir) ** 2

    if denom < 1e-6:
        raise ValueError("Rays are parallel or nearly parallel, bro!")

    diff = ray2_origin - ray1_origin
    cross_diff_dir2 = np.cross(diff, ray2_direction)
    cross_diff_dir1 = np.cross(diff, ray1_direction)

    t1 = np.dot(cross_diff_dir2, cross_dir) / denom
    t2 = np.dot(cross_diff_dir1, cross_dir) / denom

    point1 = ray1_origin + t1 * ray1_direction
    point2 = ray2_origin + t2 * ray2_direction

    midpoint = (point1 + point2) / 2.0

    return midpoint



# %%
if 'vis' not in globals():
    vis = viser.ViserServer()

# %%
from project.utils.spatial.pose import to_homogeneous_2D
from project.utils.viser_utils import show_pose, show_line_segment, show_points


t_final: np.ndarray
R_final: np.ndarray

# these are equivalent to rays of length 1 from image center in camera space
pt1 = to_homogeneous_2D(points_1_normalized[0].reshape(1, 2))
pt2 = to_homogeneous_2D(points_2_normalized[0].reshape(1, 2))



# %%
debug_1 = view_1.color.copy()
debug_2 = view_2.color.copy()

debug_1 = cv2.circle(debug_1, points_1[idx_sample][0].astype(int), radius=5, thickness=-1, color=(0, 255, 0))
debug_2 = cv2.circle(debug_2, points_2[idx_sample][0].astype(int), radius=5, thickness=-1, color=(0, 255, 0))
mediapy.show_images([debug_1, debug_2])

# %%

for t_possible, R_possible in possible_solutions:
    cam_2_in_cam_1 = Pose.from_rotmat_trans(R_possible, t_possible)

    O1 = np.zeros(3)

    d_1 = pt1


    show_pose(vis, Pose.identity(), name="cam_1")
    show_pose(vis, cam_2_in_cam_1, name="cam_2")


    O2 = t_possible
    d_2 = cam_2_in_cam_1.apply(pt2) - O2

    show_line_segment(vis, np.zeros((1, 3)), d_1.reshape(1, 3), name="ray_1")
    show_line_segment(vis, O2.reshape(1, 3), (d_2 + O2).reshape(1, 3), name="ray_2")

    print(f"{O1=}\n{d_1=}\n{O2=}\n{d_2=}")
    point_in_cam_1 = triangulate_point(O1.squeeze(), d_1.squeeze(), O2.squeeze(), d_2.squeeze())
    print(point_in_cam_1)

    if point_in_cam_1[2] < 0:
        continue


    point_in_cam_2 = cam_2_in_cam_1.inv.apply(point_in_cam_1.reshape(1, 3)).reshape(3)

    if point_in_cam_2[2] < 0:
        continue

    
    show_points(vis, point_in_cam_1.reshape(1, 3), name="P")

    print("Found")

    t_final = t_possible
    R_final = R_possible

    break




    
