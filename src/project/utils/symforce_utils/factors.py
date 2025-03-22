import symforce.symbolic as sf
from symforce import codegen

from project.utils.paths import symforce_general_factor_path


def point_reprojection_factor(
    camera_calibration: sf.LinearCameraCal,
    world_to_cam: sf.Pose3,
    landmark_loc: sf.V3,
    observation: sf.V2,
    epsilon: sf.Scalar,
) -> sf.V2:
    landmark_in_cam = world_to_cam.inverse() * landmark_loc
    landmark_reprojection, _ = camera_calibration.pixel_from_camera_point(landmark_in_cam, epsilon)
    return observation - landmark_reprojection


def generate_point_reprojection_factor() -> None:
    cg = codegen.Codegen.function(func=point_reprojection_factor, config=codegen.CppConfig())  # type: ignore

    cg = cg.with_linearization(which_args=["world_to_cam", "landmark_loc"])

    cg.generate_function(output_dir=symforce_general_factor_path(), namespace="general_factors")


def depth_factor(world_to_cam: sf.Pose3, landmark_loc: sf.V3, depth: sf.Scalar) -> sf.V1:
    landmark_in_cam = world_to_cam.inverse() * landmark_loc
    return sf.V1(depth - landmark_in_cam[2])  # type: ignore


def generate_depth_factor() -> None:
    cg = codegen.Codegen.function(func=depth_factor, config=codegen.CppConfig())  # type: ignore

    cg = cg.with_linearization(which_args=["world_to_cam", "landmark_loc"])

    cg.generate_function(output_dir=symforce_general_factor_path(), namespace="general_factors")
