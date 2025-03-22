#pragma once
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <foundation/types.hpp>

namespace py = pybind11;
namespace foundation {

Vector6d SE3_log(py::EigenDRef<Eigen::Matrix4d> mat);

Eigen::Matrix4d se3_exp(py::EigenDRef<Vector6d> se3);

}  // namespace foundation