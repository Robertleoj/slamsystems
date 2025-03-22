#pragma once
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <foundation/types.hpp>
#include <vector>

namespace py = pybind11;

namespace foundation {

std::pair<Eigen::Vector3d, Eigen::Vector3d> solve_pnp_ceres(
    const std::vector<Eigen::Vector2d> image_points,
    const std::vector<Eigen::Vector3d> object_points, const Eigen::Matrix3d K,
    const Eigen::Vector3d initial_rvec, const Eigen::Vector3d initial_tvec);

std::pair<Eigen::Vector3d, Eigen::Vector3d> solve_pnp_ceres_pywrapper(
    double_array image_points, double_array object_points, double_array K,
    double_array initial_rvec, double_array initial_tvec);

}