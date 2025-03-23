
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace foundation {
void init_pose_graph(py::module_& m);
}