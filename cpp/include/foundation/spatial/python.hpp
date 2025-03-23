#pragma once
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace foundation {
void init_spatial(py::module_& m);
}