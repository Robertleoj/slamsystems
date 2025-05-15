#pragma once

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace py = pybind11;
namespace foundation {

class CameraParams {
   public:
    CameraParams(Eigen::Matrix3d K);
    CameraParams(double fx, double fy, double cx, double cy);
    CameraParams(Eigen::Matrix3d K, Eigen::Matrix<double, 14, 1> dist_coeffs);

    CameraParams(
        py::EigenDRef<Eigen::Matrix3d> K,
        py::EigenDRef<Eigen::Matrix<double, 14, 1>> dist_coeffs,
        unsigned int width,
        unsigned int height
    );

    double fx();
    double fy();
    double cx();
    double cy();
    Eigen::Matrix3d K;
    unsigned int width, height;
    Eigen::Matrix<double, 14, 1> dist_coeffs;
};

}  // namespace foundation
