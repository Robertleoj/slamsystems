#include <Eigen/Core>
#include <Eigen/Geometry>
#include <foundation/utils/camera.hpp>

namespace foundation {

CameraParams::CameraParams(Eigen::Matrix3d K) : K(K) {}
CameraParams::CameraParams(Eigen::Matrix3d K,
                           Eigen::Matrix<double, 14, 1> dist_coeffs)
    : K(K), dist_coeffs(dist_coeffs) {}

CameraParams::CameraParams(double fx, double fy, double cx, double cy) {
  K = Eigen::Matrix3d::Identity();
  K(0, 0) = fx;
  K(0, 2) = cx;
  K(1, 1) = fy;
  K(1, 2) = cy;
}
CameraParams::CameraParams(
    py::EigenDRef<Eigen::Matrix3d> K,
    py::EigenDRef<Eigen::Matrix<double, 14, 1>> dist_coeffs, unsigned int width,
    unsigned int height)
    : K(K), dist_coeffs(dist_coeffs), width(width), height(height) {};

sym::LinearCameraCald CameraParams::to_symforce() {
  return sym::LinearCameraCald(Eigen::Vector4d({fx(), fy(), cx(), cy()}));
}

double CameraParams::fx() { return K(0, 0); }
double CameraParams::fy() { return K(1, 1); }
double CameraParams::cx() { return K(0, 2); }
double CameraParams::cy() { return K(1, 2); }

Eigen::Matrix<double, 14, 1> dist_coeffs;

}  // namespace foundation
