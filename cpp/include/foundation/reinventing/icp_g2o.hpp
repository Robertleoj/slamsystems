#include <Eigen/Core>
#include <foundation/types.hpp>
#include <vector>

namespace foundation {

Vector6d icp_g2o(const std::vector<Eigen::Vector3d> points1,
                 const std::vector<Eigen::Vector3d> points2,
                 const Eigen::Vector3d initial_rvec,
                 const Eigen::Vector3d initial_tvec);

Vector6d icp_g2o_pywrapper(double_array points1, double_array points2,
                           double_array initial_rvec,
                           double_array initial_tvec);
}  // namespace foundation