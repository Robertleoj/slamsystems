#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <foundation/depth_slam/depth_slam.hpp>
#include <foundation/utils/camera.hpp>

namespace foundation {

void init_depth_slam(
    py::module_& m
) {
    py::class_<depth_slam::CameraPose>(m, "CameraPose")
        .def(
            py::init<int, py::EigenDRef<Eigen::Matrix4d>>(),
            py::arg("frame_id"),
            py::arg("camera_pose")
        )
        .def_readonly("frame_id", &depth_slam::CameraPose::frame_id)
        .def_readonly("camera_pose", &depth_slam::CameraPose::camera_pose);

    py::class_<depth_slam::Landmark>(m, "Landmark")
        .def(
            py::init<int, py::EigenDRef<Eigen::Vector3d>>(),
            py::arg("id"),
            py::arg("loc")
        )
        .def_readonly("id", &depth_slam::Landmark::id)
        .def_readonly("loc", &depth_slam::Landmark::loc);

    py::class_<depth_slam::LandmarkProjectionObservation>(
        m, "LandmarkProjectionObservation"
    )
        .def(
            py::init<
                int,
                int,
                py::EigenDRef<Eigen::Vector2d>,
                std::optional<double>>(),
            py::arg("frame_id"),
            py::arg("landmark_id"),
            py::arg("observation"),
            py::arg("depth")
        );

    m.def(
        "depth_slam_ba",
        &depth_slam::depth_slam_ba,
        "Depth slam bundle adjustment",
        py::arg("camera_cal"),
        py::arg("camera_poses"),
        py::arg("observations"),
        py::arg("landmarks")
    );
}
}  // namespace foundation