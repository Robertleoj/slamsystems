#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <foundation/optim/g2o/edges.hpp>
#include <foundation/optim/g2o/vertices.hpp>
#include <foundation/reinventing/icp_g2o.hpp>
#include <foundation/utils/numpy.hpp>

namespace foundation {

Vector6d icp_g2o(const std::vector<Eigen::Vector3d> points1,
                 const std::vector<Eigen::Vector3d> points2,
                 const Eigen::Vector3d initial_rvec,
                 const Eigen::Vector3d initial_tvec) {
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
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

  int N = points1.size();

  for (int i = 0; i < N; i++) {
    auto point1 = points1[i];
    auto point2 = points2[i];
    auto edge = new Two3DPointsAlignmentErrorEdge(point2);

    edge->setVertex(0, vertex);
    edge->setMeasurement(point1);
    edge->setInformation(Eigen::Matrix<double, 3, 3>::Identity());
    optimizer.addEdge(edge);
  }

  optimizer.initializeOptimization();
  optimizer.optimize(10);

  return vertex->estimate().log();
}

Vector6d icp_g2o_pywrapper(double_array points1, double_array points2,
                           double_array initial_rvec,
                           double_array initial_tvec) {
  if (points1.shape(0) != points2.shape(0)) {
    throw std::runtime_error("Must have the same length");
  }

  auto points1_eig = array_to_3d_points(points1);
  auto points2_eig = array_to_3d_points(points2);

  auto rvec_eig = array_to_3d_point(initial_rvec);
  auto tvec_eig = array_to_3d_point(initial_tvec);

  return icp_g2o(points1_eig, points2_eig, rvec_eig, tvec_eig);
}

}  // namespace foundation