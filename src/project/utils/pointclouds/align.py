from typing import cast

import einops
import numpy as np
import pypose as pp
import torch
from pypose.optim.scheduler import StopOnPlateau

from project.utils.pypose_utils import pose_from_pypose
from project.utils.spatial import Pose


def align_point_matches_svd(points1: np.ndarray, points2: np.ndarray) -> Pose:
    assert points1.shape[0] == points2.shape[0]
    assert len(points1.shape) == len(points2.shape) == 2
    assert points1.shape[1] == points1.shape[1] == 3

    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)

    decentroids1 = points1 - centroid1
    decentroids2 = points2 - centroid2

    W = einops.einsum(decentroids1, decentroids2, "n d1, n d2 -> d1 d2")

    if np.linalg.matrix_rank(W) != 3:
        raise ValueError("W does not have full rank")

    U, _S, V_T = np.linalg.svd(W)

    R = U @ V_T

    if abs(np.linalg.det(R) - 1) > 1e-3:
        R = -R

    t = centroid1.reshape(3, 1) - R @ centroid2.reshape(3, 1)

    return Pose.from_rotmat_trans(R, t.squeeze())


def align_point_matches_svd_ransac(
    points1: np.ndarray, points2: np.ndarray, inlier_threshold: float, num_tries: int = 20, sample_size: int = 5
) -> tuple[Pose, np.ndarray]:
    best_num_inliers = -1
    best_inlier_mask = None
    best_pose = None

    for _ in range(num_tries):
        indices = np.random.choice(len(points1), size=sample_size, replace=False)

        p1 = points1[indices]
        p2 = points2[indices]

        try:
            pose = align_point_matches_svd(p1, p2)
        except ValueError as e:
            print(e)
            continue

        points2_transformed = pose.apply(points2)

        distances = np.linalg.norm(points1 - points2_transformed, axis=1)
        inlier_mask = cast(np.ndarray, distances < inlier_threshold)
        num_inliers = inlier_mask.sum()

        if num_inliers > best_num_inliers:
            best_pose = pose
            best_inlier_mask = inlier_mask
            best_num_inliers = num_inliers

    assert best_pose is not None
    assert best_inlier_mask is not None

    return best_pose, best_inlier_mask


def align_point_matches_pypose(points1: np.ndarray, points2: np.ndarray) -> Pose:
    class ICPPoseGraph(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            init = pp.identity_SE3()
            self.pose = pp.Parameter(init)

        def forward(self, points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
            points2_trans = self.pose @ points2

            return points1 - points2_trans

    pnp_g = ICPPoseGraph()
    solver = pp.optim.solver.Cholesky()
    strategy = pp.optim.strategy.TrustRegion()

    optimizer = pp.optim.LM(pnp_g, solver=solver, strategy=strategy, min=1e-6)
    scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=True)

    p1_t = torch.from_numpy(points1).to(dtype=torch.float32)
    p2_t = torch.from_numpy(points2).to(dtype=torch.float32)

    scheduler.optimize(input=(p1_t, p2_t))

    return pose_from_pypose(pnp_g.pose)
