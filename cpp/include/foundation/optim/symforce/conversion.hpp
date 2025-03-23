#pragma once
#include <sym/pose3.h>

#include <Eigen/Core>

namespace foundation {

sym::Pose3d pose_from_homo_mat(Eigen::Matrix4d& mat);
}  // namespace foundation