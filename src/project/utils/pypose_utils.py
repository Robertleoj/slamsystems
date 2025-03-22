"""Utilities for using the pypose library

https://pypose.org/
https://github.com/pypose/pypose
"""

import pypose as pp

from project.utils.spatial import Pose, Rotation


def pose_from_pypose(pp_pose: pp.LieTensor) -> Pose:
    assert pp.function.checking.is_SE3(pp_pose)

    pose_vec = pp_pose.detach().numpy().squeeze()
    trans_vec = pose_vec[:3]
    xyzw = pose_vec[3:]

    rot = Rotation.from_xyzw(xyzw)
    return Pose.from_rot_trans(rot, trans_vec)
