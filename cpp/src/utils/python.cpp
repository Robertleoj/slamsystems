#include <pybind11/eigen.h>
#include <format>
#include <foundation/utils/camera.hpp>
#include <foundation/utils/python.hpp>

namespace foundation {
void init_utils(py::module_& m) {
  py::class_<CameraParams>(m, "CameraParams")
      .def(py::init<double, double, double, double>(), py::arg("fx"),
           py::arg("fy"), py::arg("cx"), py::arg("cy"))
      .def(py::init<py::EigenDRef<Eigen::Matrix3d>,
                    py::EigenDRef<Eigen::Matrix<double, 14, 1>>, unsigned int,
                    unsigned int>(),
           py::arg("K"), py::arg("dist_coeffs"), py::arg("width"),
           py::arg("height"))
      .def("__repr__", [](CameraParams& cam) {
        return std::format(
            "CameraParams(fx={}, fy={}, cx={}, cy={}, width={}, height={})",
            cam.fx(), cam.fy(), cam.cx(), cam.cy(), cam.width, cam.height);
      });
}
}  // namespace foundation