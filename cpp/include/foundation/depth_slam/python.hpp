#pragma once
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace foundation {
void init_depth_slam(py::module_& m);
}