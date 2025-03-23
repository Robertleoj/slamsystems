#include <foundation/spatial/lie_algebra.hpp>
#include <foundation/spatial/python.hpp>

namespace foundation {

void init_spatial(py::module_& m) {
  m.def("SE3_log", &SE3_log, "Logarithm mapping from SE3 to se3",
        py::arg("mat"));

  m.def("se3_exp", &se3_exp, "Exponential mapping from se3 to SE3",
        py::arg("se3"));
}
}  // namespace foundation