from pathlib import Path
from typing import cast

import symforce.symbolic as sf
from symforce import codegen

from project.utils.markers import get_corners_in_tag


def tag_reprojection_error(
    camera: sf.LinearCameraCal,
    cam_pose: sf.Pose3,
    tag_pose: sf.Pose3,
    measurement: sf.V8,
    tag_side_length: sf.Scalar,
    epsilon: sf.Scalar,
):
    observation = sf.V8()
    for i, tag_corner in enumerate(get_corners_in_tag(tag_side_length)):
        corner_sf = sf.V3(tag_corner)

        corner_in_world = cast(sf.V3, tag_pose * corner_sf)
        corner_in_cam = cast(sf.V3, cam_pose.inverse() * corner_in_world)

        tag_pixels, _ = camera.pixel_from_camera_point(corner_in_cam, epsilon)

        observation[2 * i : 2 * (i + 1)] = tag_pixels

    return measurement - observation


def generate_code(output_dir: Path, namespace: str = "tag_slam_factors") -> None:
    cg = codegen.Codegen.function(func=tag_reprojection_error, config=codegen.CppConfig())  # type: ignore

    cg = cg.with_linearization(which_args=["cam_pose", "tag_pose"])

    cg.generate_function(output_dir=output_dir, namespace=namespace)
