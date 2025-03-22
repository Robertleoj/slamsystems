"""Utilities for the rerun library.

Python API:
https://ref.rerun.io/docs/python/0.22.1/common/

General Docs:
https://rerun.io/docs/
"""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from project.utils.camera import Camera, Intrinsics
from project.utils.colors import Color, Colors
from project.utils.jupyter import in_jupyter
from project.utils.spatial import Pose, Vec3

RR_INITIALIZED = False


def label3D(text: str, color: Color = Colors.WHITE, location: Vec3 = Vec3.zero()) -> rr.Points3D:
    return rr.Points3D(positions=[location.to_arr()], radii=[[0]], labels=[text], show_labels=True, colors=[color])


def triad(scale: float = 50.0):
    vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * scale

    return rr.Arrows3D(
        vectors=vecs,
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    )


def transform(pose: Pose):
    return rr.Transform3D(mat3x3=pose.rot_mat, translation=pose.tvec)


def should_cancel_init() -> bool:
    return in_jupyter() and RR_INITIALIZED


def set_initialized() -> None:
    global RR_INITIALIZED
    RR_INITIALIZED = True


def init(app_name: str = "app") -> None:
    """Wrapper function around init to avoid double inits in notebooks"""
    if should_cancel_init():
        return

    rr.init(app_name, spawn=True)
    set_initialized()


def simple_init(app_name: str = "app"):
    init(app_name)
    rr.send_blueprint(rrb.Blueprint(collapse_panels=True))


def log_reference_frame(path: str, pose: Pose, label: str | None = None):
    rr.log(path, transform(pose))
    rr.log(f"{path}/triad", triad())
    if label is not None:
        rr.log(f"{path}/label", label3D("rvectvec"))


def log_trajectory(path: str, traj: list[Pose], traj_color: Color) -> None:
    for i, pose in enumerate(traj):
        log_reference_frame(f"{path}/pose{i}", pose)

    translations = np.array([p.tvec for p in traj])

    rr.log(f"{path}/line", rr.LineStrips3D(strips=[translations], colors=[traj_color]))


def camera(intrinsics: Intrinsics, scale: float = 50.0) -> rr.Pinhole:
    return rr.Pinhole(
        image_from_camera=intrinsics.camera_matrix,
        resolution=[intrinsics.width, intrinsics.height],
        image_plane_distance=scale,
    )


def log_camera(
    camera_space_path: str,
    cam: Camera,
    image: np.ndarray | None = None,
    depth_image: np.ndarray | None = None,
    scale: float = 50.0,
    undistort: bool = True,
    point_fill_ratio: float = 1.0,
    static: bool = False,
) -> None:
    rr.log(camera_space_path, transform(cam.extrinsics), static=static)

    rr.log(f"{camera_space_path}/cam", camera(cam.intrinsics, scale=scale), static=static)

    if image is not None:
        if undistort:
            image = cam.intrinsics.undistort_image(image)
        rr.log(f"{camera_space_path}/cam/img", rr.Image(image), static=static)

    if depth_image is not None:
        if undistort:
            depth_image = cam.intrinsics.undistort_image(depth_image)
        rr.log(
            f"{camera_space_path}/cam/depth",
            rr.DepthImage(depth_image, point_fill_ratio=point_fill_ratio),
            static=static,
        )


def image_tabs_bp(root_path: str, img_names: list[str], tab_name: str = "images") -> rrb.Tabs:
    return rrb.Tabs(*[rrb.Spatial2DView(origin=f"{root_path}/{img_name}") for img_name in img_names], name=tab_name)


def log_point_matches(root_path: str, points1: np.ndarray, points2: np.ndarray):
    rr.log(f"{root_path}/points1", rr.Points3D(points1, radii=3.0, colors=Colors.RED))
    rr.log(f"{root_path}/points2", rr.Points3D(points2, radii=3.0, colors=Colors.GREEN))
    rr.log(f"{root_path}/point_connections", rr.Arrows3D(vectors=points2 - points1, origins=points1))
