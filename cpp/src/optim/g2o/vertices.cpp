#include <foundation/optim/g2o/vertices.hpp>
#include <foundation/types.hpp>

namespace foundation {

void CameraPose::setToOriginImpl() { _estimate = Sophus::SE3d(); }

void CameraPose::oplusImpl(const double *update) {
  Vector6d update_eigen;
  for (int i = 0; i < 6; i++) {
    update_eigen(i, 0) = update[i];
  }

  Sophus::SE3d update_pose = Sophus::SE3d::exp(update_eigen);

  _estimate = update_pose * _estimate;
}

bool CameraPose::read(std::istream &in) { return false; }

bool CameraPose::write(std::ostream &out) const { return false; }

}