import numpy as np

from project.utils.spatial.pose import Pose


def test_inverse():
    for _ in range(10):
        p = Pose.random()
        assert np.allclose((p @ p.inv).mat, np.eye(4))


def test_exp_log() -> None:
    for _ in range(10):
        p = Pose.random()
        se3 = p.log()
        p2 = Pose.exp(se3)
        assert np.allclose(p.mat, p2.mat)


def test_rvec_tvec() -> None:
    for _ in range(10):
        rvec = np.random.randn(3)
        tvec = np.random.randn(3)

        pose = Pose.from_rvec_tvec(rvec, tvec)

        rvec = pose.rvec
        tvec = pose.tvec

        assert np.allclose(pose.rvec, rvec)
        assert np.allclose(pose.tvec, tvec)
