#pragma once

#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

namespace py = pybind11;
namespace foundation {

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef py::array_t<double, py::array::c_style> double_array;
typedef py::array_t<uint8_t, py::array::c_style> uint8_array;

}  // namespace foundation
