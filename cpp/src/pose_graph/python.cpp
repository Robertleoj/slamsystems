#include <pybind11/eigen.h>
#include <foundation/pose_graph/pose_graph.hpp>
#include <foundation/pose_graph/python.hpp>

namespace foundation {

void init_pose_graph(
    py::module_& m
) {
    py::class_<PoseGraphEdge>(m, "PoseGraphEdge")
        .def(
            py::init<int, int, py::EigenDRef<Eigen::Matrix4d>>(),
            py::arg("v1_id"),
            py::arg("v2_id"),
            py::arg("v1_to_v2")
        );
    py::class_<PoseGraphVertex>(m, "PoseGraphVertex")
        .def(
            py::init<int, py::EigenDRef<Eigen::Matrix4d>>(),
            py::arg("id"),
            py::arg("pose")
        );

    m.def(
        "pose_graph_ba",
        &pose_graph_ba,
        "Pose graph BA with symforce",
        py::arg("vertices"),
        py::arg("edges")
    );
}

}  // namespace foundation