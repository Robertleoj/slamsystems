#include <foundation/optim/symforce/conversion.hpp>

namespace foundation {

sym::Pose3d pose_from_homo_mat(Eigen::Matrix4d &mat) {
  Eigen::Quaterniond q(
      mat.block<3, 3>(0, 0));                 // Extract rotation as quaternion
  Eigen::Vector3d t = mat.block<3, 1>(0, 3);  // Extract translation

  return sym::Pose3d({q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z()});
}
}  // namespace foundation