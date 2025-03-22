#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <foundation/reinventing/icp_ceres.hpp>
#include <foundation/utils/numpy.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

namespace foundation {

struct PointDiffCost {
 public:
  PointDiffCost(const Eigen::Vector3d &point1, const Eigen::Vector3d &point2)
      : _point1(point1), _point2(point2) {}

  template <typename T>
  bool operator()(const T *const camera, T *residuals) const {
    T point1[3] = {T(_point1.x()), T(_point1.y()), T(_point1.z())};
    T point2[3] = {T(_point2.x()), T(_point2.y()), T(_point2.z())};

    T point2_transformed[3];
    ceres::AngleAxisRotatePoint(camera, point2, point2_transformed);

    for (int i = 0; i < 3; i++) {
      point2_transformed[i] += camera[i + 3];
    }

    for (int i = 0; i < 3; i++) {
      residuals[i] = point1[i] - point2_transformed[i];
    }
    return true;
  }

 private:
  const Eigen::Vector3d _point1;
  const Eigen::Vector3d _point2;
};

Vector6d icp_ceres(const std::vector<Eigen::Vector3d> points1,
                   const std::vector<Eigen::Vector3d> points2,
                   const Eigen::Vector3d initial_rvec,
                   const Eigen::Vector3d initial_tvec) {
  double camera[6];
  for (int i = 0; i < 3; i++) {
    camera[i] = initial_rvec[i];
    camera[i + 3] = initial_tvec[i];
  }

  ceres::Problem problem;

  for (int i = 0; i < points1.size(); i++) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<PointDiffCost, 3, 6>(
            new PointDiffCost(points1[i], points2[i])),
        nullptr, (double *)camera);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  Eigen::Vector3d rvec(camera[0], camera[1], camera[2]);
  Eigen::Vector3d tvec(camera[3], camera[4], camera[5]);

  return Sophus::SE3d(Sophus::SO3d::exp(rvec), tvec).log();
}

Vector6d icp_ceres_pywrapper(double_array points1, double_array points2,
                             double_array initial_rvec,
                             double_array initial_tvec) {
  auto points1_eig = array_to_3d_points(points1);
  auto points2_eig = array_to_3d_points(points2);
  auto rvec_eig = array_to_3d_point(initial_rvec);
  auto tvec_eig = array_to_3d_point(initial_tvec);

  return icp_ceres(points1_eig, points2_eig, rvec_eig, tvec_eig);
}

}  // namespace foundation