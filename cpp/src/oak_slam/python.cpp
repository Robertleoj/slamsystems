#include <foundation/oak_slam/oak_slam.hpp>
#include <foundation/oak_slam/python.hpp>

namespace foundation {
namespace oak_slam {

void init_oak_slam(
    py::module_& m
) {
    py::class_<OakSlam>(m, "OakSlam")
        .def(
            py::init<
                CameraParams&,
                CameraParams&,
                CameraParams&,
                py::EigenDRef<Eigen::Matrix4d>,
                py::EigenDRef<Eigen::Matrix4d>>(),
            py::arg("color_intrinsics"),
            py::arg("left_intrinsics"),
            py::arg("right_intrinsics"),
            py::arg("center_to_left"),
            py::arg("center_to_right")
        )
        .def(
            "process_frame",
            py::overload_cast<img_array, img_array, img_array>(
                &OakSlam::process_frame
            ),
            py::arg("center_color"),
            py::arg("left_mono"),
            py::arg("right_mono")
        );
}

}  // namespace oak_slam
}  // namespace foundation
