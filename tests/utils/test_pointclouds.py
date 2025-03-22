import numpy as np

from project.utils.pointclouds.align import align_point_matches_svd, align_point_matches_svd_ransac
from project.utils.spatial.pose import Pose


def test_align_point_matches_svd():
    points1 = np.random.randn(50, 3)
    points2_aligned = points1.copy()

    random_pose = Pose.random()

    points2 = random_pose.apply(points2_aligned)

    pose = align_point_matches_svd(points1, points2)

    points2_recovered = pose.apply(points2)

    assert np.allclose(points1, points2_recovered)


def test_align_point_matches_svd_ransac():
    points1 = np.random.randn(50, 3)
    points2_aligned = points1.copy()

    points2_aligned[:5] += np.random.randn(5, 3) * 10

    random_pose = Pose.random()

    points2 = random_pose.apply(points2_aligned)

    pose, inlier_mask = align_point_matches_svd_ransac(points1, points2, 3, num_tries=100)

    points2_recovered = pose.apply(points2)

    assert (~inlier_mask).sum() == 5
    assert np.allclose(points1[5:], points2_recovered[5:], rtol=1e-3)
