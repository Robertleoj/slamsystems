#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace foundation {
namespace oak_slam {

void init_oak_slam(py::module_& m);

}  // namespace oak_slam
}  // namespace foundation