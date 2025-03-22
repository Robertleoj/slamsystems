from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from einops import einsum
from scipy.spatial.transform import Rotation as R

from project.foundation.spatial import SE3_log, se3_exp


@dataclass(frozen=True)
class Pose:
    mat: np.ndarray

    def snap(self) -> Pose:
        """Snaps a 4x4 homogeneous transformation matrix to ensure:
        - The rotation part is a valid SO(3) matrix (orthogonal, det(R) = 1)
        - The translation part remains unchanged

        Args:
            T (numpy.ndarray): 4x4 transformation matrix

        Returns:
            numpy.ndarray: Snapped 4x4 transformation matrix
        """
        R = self.rot_mat

        # Perform SVD to correct R
        U, _, Vt = np.linalg.svd(R)
        R_fixed = U @ Vt  # Project onto closest SO(3)

        assert np.linalg.det(R_fixed) > 0, "nonpositive determinant!"

        return Pose.from_rotmat_trans(R_fixed, self.tvec)

    def __repr__(self):
        euler = self.rot.scipy_rot.as_euler("xyz", degrees=True)
        trans = self.tvec
        return f"Pose('xyz'=[{euler[0]}, {euler[1]}, {euler[2]}], t=[{trans[0]}, {trans[1]}, {trans[2]}])"

    def __matmul__(self, other: Pose) -> Pose:
        return Pose(self.mat @ other.mat)

    @property
    def rot_mat(self) -> np.ndarray:
        return self.mat[:3, :3].copy()

    @property
    def rot(self) -> Rotation:
        return Rotation(self.rot_mat)

    @property
    def inv(self) -> Pose:
        t = self.tvec.reshape(3, 1)

        inv = np.eye(4)

        R_T = self.rot.mat.T
        inv[:3, :3] = R_T
        inv[:3, 3] = (-R_T @ t).flatten()

        return Pose(inv)

    @property
    def rvec(self) -> np.ndarray:
        return self.rot.rvec

    @property
    def tvec(self) -> np.ndarray:
        return self.mat[:3, 3].copy().flatten()

    @staticmethod
    def identity() -> Pose:
        return Pose(np.eye(4))

    @staticmethod
    def random(scale: float = 0.1) -> Pose:
        return Pose.exp(np.random.randn(6) * scale)

    @staticmethod
    def from_rotmat_trans(rot_mat: np.ndarray, trans: np.ndarray) -> Pose:
        assert rot_mat.shape == (3, 3)
        assert trans.shape == (3,), f"Trans has shape {trans.shape}"
        assert np.allclose(rot_mat.T @ rot_mat, np.eye(3)), f"R={rot_mat}"

        T = np.eye(4)
        T[:3, :3] = rot_mat
        T[:3, 3] = trans

        return Pose(T)

    @staticmethod
    def from_rot_trans(rot: Rotation, trans: np.ndarray) -> Pose:
        return Pose.from_rotmat_trans(rot.mat, trans)

    @staticmethod
    def from_xyzw_trans(xyzw: np.ndarray, trans: np.ndarray) -> Pose:
        rot = Rotation.from_xyzw(xyzw)
        return Pose.from_rot_trans(rot, trans)

    @staticmethod
    def from_wxyz_trans(wxyz: np.ndarray, trans: np.ndarray) -> Pose:
        rot = Rotation.from_wxyz(wxyz)
        return Pose.from_rot_trans(rot, trans)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform points with a left-multiply: Tp

        Args:
            points: N x 3 array

        Returns:
            transformed_points: N x 3 array

        """
        homo = to_homogeneous_3D(points)

        transformed = einsum(self.mat, homo, "h d, n d -> n h")

        return to_inhomogeneous_3D(transformed)

    def with_translation(self, translation: np.ndarray) -> Pose:
        return Pose.from_rotmat_trans(self.rot_mat, translation)

    def scale_translation(self, scale: float) -> Pose:
        return Pose.from_rotmat_trans(self.rot_mat, self.tvec * scale)

    @staticmethod
    def from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> Pose:
        rot_mat = R.from_rotvec(rvec).as_matrix()

        mat = np.eye(4)
        mat[:3, :3] = rot_mat
        mat[:3, 3] = tvec
        return Pose(mat)

    def log(self, rot_first: bool = False) -> np.ndarray:
        """Logarithmic mapping to se3

        Returns:
            se3: shape (6,), contains [rho, phi], where rho is the translational
                part, and phi is the rotational part, equal to rvec.

        """
        vc = SE3_log(self.mat)

        if not rot_first:
            return vc

        return np.roll(vc, 3)

    @staticmethod
    def exp(se3: np.ndarray, rot_first: bool = False) -> Pose:
        """Exponentiate a se3 vector into SE3.

        Args:
            se3: shape (6,) vector, should be of the form [rho, phi],
                hwhere rho is the translational part and phi is the rotational part.
        """
        se3 = se3.squeeze()
        assert se3.shape == (6,), f"Got shape {se3.shape}"

        if rot_first:
            # rotations are first, but we want translation first
            se3 = np.roll(se3, 3)

        return Pose(se3_exp(se3))


@dataclass(frozen=True)
class Rotation:
    # 3 x 3 rotation matrix
    mat: np.ndarray

    def __matmul__(self, other: Rotation) -> Rotation:
        return Rotation(self.mat @ other.mat)

    @staticmethod
    def from_rvec(rvec: np.ndarray) -> Rotation:
        rvec = rvec.squeeze()
        assert rvec.shape == (3,)

        mat = R.from_rotvec(rvec).as_matrix()

        return Rotation(mat)

    @property
    def scipy_rot(self) -> R:
        return R.from_matrix(self.mat)

    @staticmethod
    def from_scipy_rot(scipy_rot: R) -> Rotation:
        return Rotation(scipy_rot.as_matrix())

    @property
    def rvec(self) -> np.ndarray:
        return self.scipy_rot.as_rotvec()

    def log(self) -> np.ndarray:
        return self.rvec

    @staticmethod
    def exp(so3: np.ndarray) -> Rotation:
        return Rotation.from_rvec(so3)

    @property
    def inv(self) -> Rotation:
        return Rotation(self.mat.T)

    @property
    def wxyz(self) -> np.ndarray:
        x, y, z, w = self.xyzw
        return np.array([w, x, y, z])

    @property
    def xyzw(self) -> np.ndarray:
        return self.scipy_rot.as_quat(canonical=True)

    @staticmethod
    def from_xyzw(xyzw: np.ndarray) -> Rotation:
        assert xyzw.shape == (4,)
        return Rotation.from_scipy_rot(R.from_quat(xyzw))

    @staticmethod
    def from_wxyz(wxyz: np.ndarray) -> Rotation:
        assert wxyz.shape == (4,)
        w, x, y, z = wxyz
        return Rotation.from_xyzw(np.array([x, y, z, w]))


def to_homogeneous_3D(points: np.ndarray) -> np.ndarray:
    if points.shape[1] == 4:
        return points

    out = np.ones((points.shape[0], 4))
    out[:, :3] = points
    return out


def to_inhomogeneous_3D(points: np.ndarray) -> np.ndarray:
    if points.shape[1] not in (3, 4):
        raise ValueError("invalid points")

    if points.shape[1] == 3:
        return points

    out = points.copy()[:, :3]
    out /= points[:, 3, None]

    return out


def to_homogeneous_2D(points: np.ndarray) -> np.ndarray:
    if points.shape[1] == 3:
        return points

    out = np.ones((points.shape[0], 3))
    out[:, :2] = points
    return out


def to_inhomogeneous_2D(points: np.ndarray) -> np.ndarray:
    if points.shape[1] == 2:
        return points

    out = points.copy()[:, :2]
    out /= points[:, 2]

    return out


def rx_R(angle: float, degrees=False) -> np.ndarray:
    return R.from_euler("x", angle, degrees=degrees).as_matrix()


def ry_R(angle: float, degrees=False) -> np.ndarray:
    return R.from_euler("y", angle, degrees=degrees).as_matrix()


def rz_R(angle: float, degrees=False) -> np.ndarray:
    return R.from_euler("z", angle, degrees=degrees).as_matrix()


def skew(x: np.ndarray):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
