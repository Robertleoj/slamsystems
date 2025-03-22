from typing import Literal

import numpy as np
import viser

from project.utils.camera import Camera
from project.utils.spatial import Pose

MM_SCALE = 100


def _scale_camera(camera: Camera) -> Camera:
    return Camera(intrinsics=camera.intrinsics, extrinsics=camera.extrinsics.scale_translation(1 / MM_SCALE))


def _scale_pose(pose: Pose) -> Pose:
    return pose.scale_translation(1 / MM_SCALE)


def show_pose(vis: viser.ViserServer, pose: Pose, name: str = "pose/frame") -> None:
    vis.scene.add_frame(name, wxyz=pose.rot.wxyz, position=pose.tvec)


def show_line_segments(
    vis: viser.ViserServer,
    starts: np.ndarray,
    ends: np.ndarray,
    thickness: float = 1.0,
    name: str = "lines",
    colors: tuple[int, int, int] | np.ndarray = (255, 0, 0),
    apply_scale: bool = True,
) -> None:
    if apply_scale:
        starts = starts / MM_SCALE
        ends = ends / MM_SCALE

    vis.scene.add_line_segments(points=np.stack([starts, ends], axis=1), name=name, colors=colors, line_width=thickness)


def show_points(
    vis: viser.ViserServer,
    points: np.ndarray,
    colors: tuple[int, int, int] | np.ndarray = (0, 255, 0),
    name: str = "points",
    point_size: float = 0.05,
    point_shape: Literal["circle", "square"] = "circle",
    apply_scale: bool = True,
) -> None:
    assert len(points.shape) == 2 and points.shape[1] == 3

    if apply_scale:
        points = points / MM_SCALE

    vis.scene.add_point_cloud(name=name, points=points, colors=colors, point_size=point_size, point_shape=point_shape)


def show_camera(
    vis: viser.ViserServer, camera: Camera, name="cam", image: np.ndarray | None = None, apply_scale: bool = True
) -> None:
    if apply_scale:
        camera = _scale_camera(camera)

    show_pose(vis, camera.extrinsics, name=name)

    if image is not None:
        image = camera.intrinsics.undistort_image(image)

    vis.scene.add_camera_frustum(
        name=f"{name}/frust", fov=camera.intrinsics.fov()[1], aspect=camera.intrinsics.aspect_ratio(), image=image
    )


def show_projection_rays(
    vis: viser.ViserServer,
    points_in_cam: np.ndarray,
    name="proj_rays",
    camera_pose: Pose = Pose.identity(),
    thickness: float = 1.0,
    colors: tuple[int, int, int] | np.ndarray = (255, 0, 0),
    apply_scale: bool = True,
) -> None:
    if apply_scale:
        points_in_cam = points_in_cam / MM_SCALE
        camera_pose = _scale_pose(camera_pose)

    N = points_in_cam.shape[0]
    starts = np.zeros((N, 3))

    starts[:] = camera_pose.tvec

    ends = camera_pose.apply(points_in_cam)

    vis.scene.add_line_segments(points=np.stack([starts, ends], axis=1), name=name, colors=colors, line_width=thickness)
