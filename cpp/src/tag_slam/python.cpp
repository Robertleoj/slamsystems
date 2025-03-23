#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <foundation/tag_slam/python.hpp>
#include <foundation/tag_slam/tag_slam.hpp>

namespace foundation {
void init_tag_slam(
    py::module_& m
) {
    py::class_<tag_slam::CameraPose>(m, "CameraPose")
        .def(
            py::init<int, py::EigenDRef<Eigen::Matrix4d>>(),
            py::arg("frame_id"),
            py::arg("camera_pose")
        )
        .def_readonly("frame_id", &tag_slam::CameraPose::frame_id)
        .def_readonly("pose", &tag_slam::CameraPose::camera_pose);

    py::class_<tag_slam::Tag>(m, "Tag")
        .def(
            py::init<int, py::EigenDRef<Eigen::Matrix4d>>(),
            py::arg("tag_id"),
            py::arg("tag_pose")
        )
        .def_readonly("tag_id", &tag_slam::Tag::tag_id)
        .def_readonly("pose", &tag_slam::Tag::tag_pose);

    py::class_<tag_slam::TagObservation>(m, "TagObservation")
        .def(
            py::init<int, int, py::EigenDRef<Eigen::Matrix<double, 8, 1>>>(),
            py::arg("frame_id"),
            py::arg("tag_id"),
            py::arg("observation")
        );

    m.def(
        "tag_slam_ba",
        &tag_slam::tag_slam_ba,
        "Tag slam bundle adjustment",
        py::arg("fxfycxcy"),
        py::arg("camera_poses"),
        py::arg("observations"),
        py::arg("tags"),
        py::arg("tag_side_length")
    );
}  // namespace init_tag_slam(py::module_
}  // namespace foundation