#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <pybind11/eigen.h>

#include <Eigen/Core>
#include <foundation/reinventing/solve_pnp_ceres.hpp>
#include <foundation/utils/numpy.hpp>
#include <iostream>
#include <sophus/se3.hpp>
#include <vector>

namespace py = pybind11;
namespace foundation {
struct ReprojectionError {
  ReprojectionError(const Eigen::Vector2d &observed,
                    const Eigen::Vector3d &point3D, const Eigen::Matrix3d &K)
      : observed_(observed), point3D_(point3D), K_(K) {}

  template <typename T>
  bool operator()(const T *const camera, T *residuals) const {
    T p3D[3] = {T(point3D_.x()), T(point3D_.y()), T(point3D_.z())};
    T p[3];

    ceres::AngleAxisRotatePoint(camera, p3D, p);
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    T u = T(K_(0, 0)) * xp + T(K_(0, 2));
    T v = T(K_(1, 1)) * yp + T(K_(1, 2));

    residuals[0] = u - T(observed_.x());
    residuals[1] = v - T(observed_.y());
    return true;
  }

 private:
  const Eigen::Vector2d observed_;
  const Eigen::Vector3d point3D_;
  const Eigen::Matrix3d K_;
};

std::pair<Eigen::Vector3d, Eigen::Vector3d> solve_pnp_ceres(
    const std::vector<Eigen::Vector2d> image_points,
    const std::vector<Eigen::Vector3d> object_points, const Eigen::Matrix3d K,
    const Eigen::Vector3d initial_rvec, const Eigen::Vector3d initial_tvec) {
  double camera[6];

  auto se3 =
      Sophus::SE3d(Sophus::SO3d::exp(initial_rvec).matrix(), initial_tvec)
          .log();

  // Pack initial guess into camera array
  for (int i = 0; i < 6; i++) {
    camera[i] = se3(i);
  }

  ceres::Problem problem;

  for (size_t i = 0; i < image_points.size(); ++i) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6>(
            new ReprojectionError(image_points[i], object_points[i], K)),
        nullptr, (double *)camera);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  Eigen::Vector3d rvec(camera[0], camera[1], camera[2]);
  Eigen::Vector3d tvec(camera[3], camera[4], camera[5]);

  return {rvec, tvec};
}
std::pair<Eigen::Vector3d, Eigen::Vector3d> solve_pnp_ceres_pywrapper(
    double_array image_points, double_array object_points, double_array K,
    double_array initial_rvec, double_array initial_tvec) {
  if (image_points.shape(0) != object_points.shape(0)) {
    throw std::runtime_error("Must have the same length");
  }

  auto image_points_eig = array_to_2d_points(image_points);
  auto object_points_eig = array_to_3d_points(object_points);

  auto K_eig = array_to_3mat(K);
  auto rvec_eig = array_to_3d_point(initial_rvec);
  auto tvec_eig = array_to_3d_point(initial_tvec);

  return solve_pnp_ceres(image_points_eig, object_points_eig, K_eig, rvec_eig,
                         tvec_eig);
}
}  // namespace foundation