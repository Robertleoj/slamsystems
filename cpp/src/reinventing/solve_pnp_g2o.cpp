#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <foundation/optim/g2o/edges.hpp>
#include <foundation/optim/g2o/vertices.hpp>
#include <foundation/reinventing/solve_pnp_g2o.hpp>
#include <foundation/utils/numpy.hpp>

namespace foundation {

std::pair<Eigen::Vector3d, Eigen::Vector3d> solve_pnp_g2o(
    const std::vector<Eigen::Vector2d> image_points,
    const std::vector<Eigen::Vector3d> object_points, const Eigen::Matrix3d K,
    const Eigen::Vector3d initial_rvec, const Eigen::Vector3d initial_tvec) {
  if (image_points.size() != object_points.size()) {
    throw std::runtime_error(
        "Number of world points must match number of image points");
  }

  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>> BlockSolverType;
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
      LinearSolverType;

  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
      std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));

  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(true);

  CameraPose *vertex = new CameraPose();

  auto rotation_matrix = Sophus::SO3d::exp(initial_rvec).matrix();

  Sophus::SE3d initial_estimate(rotation_matrix, initial_tvec);

  vertex->setEstimate(initial_estimate);
  vertex->setId(0);
  optimizer.addVertex(vertex);

  int N = image_points.size();

  CameraParams cam(K);
  for (int i = 0; i < N; i++) {
    auto image_point = image_points[i];
    auto object_point = object_points[i];
    auto edge = new Fixed3DReprojectionErrorEdge(object_point, cam);

    edge->setVertex(0, vertex);
    edge->setMeasurement(image_point);
    edge->setInformation(Eigen::Matrix<double, 2, 2>::Identity());
    optimizer.addEdge(edge);
  }

  optimizer.initializeOptimization();
  optimizer.optimize(10);

  auto camPose = vertex->estimate();

  auto rvec = camPose.so3().log();
  auto tvec = camPose.translation();

  return std::make_pair(rvec, tvec);
}
std::pair<Eigen::Vector3d, Eigen::Vector3d> solve_pnp_g2o_pywrapper(
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

  return solve_pnp_g2o(image_points_eig, object_points_eig, K_eig, rvec_eig,
                       tvec_eig);
}
}  // namespace foundation
