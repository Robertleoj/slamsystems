#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <foundation/spatial/lie_algebra.hpp>
#include <foundation/types.hpp>
#include <sophus/se3.hpp>

namespace py = pybind11;

namespace foundation {

Vector6d SE3_log(py::EigenDRef<Eigen::Matrix4d> mat) {
  Sophus::SE3d pose(mat);

  return pose.log();
}

Eigen::Matrix4d se3_exp(py::EigenDRef<Vector6d> se3) {
  return Sophus::SE3d::exp(se3).matrix();
}

}  // namespace foundation