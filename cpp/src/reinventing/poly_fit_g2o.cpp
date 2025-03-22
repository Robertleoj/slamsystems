#include <g2o/core/base_variable_sized_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <Eigen/Core>
#include <foundation/reinventing/poly_fit_g2o.hpp>
#include <iostream>
#include <vector>

namespace foundation {
class PolynomialCoefficient : public g2o::BaseVertex<1, double> {
 public:
  virtual void setToOriginImpl() override { _estimate = 0.0; }

  virtual void oplusImpl(const double *update) override {
    _estimate += update[0];
  }

  virtual bool read(std::istream &in) { return false; }
  virtual bool write(std::ostream &out) const { return false; }
};

class PolyFitEdge : public g2o::BaseVariableSizedEdge<1, double> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PolyFitEdge(double x, int order)
      : BaseVariableSizedEdge(), _x(x), _order(order) {
    resize(order + 1);
  }

  virtual void computeError() override {
    double y_est = 0;

    for (int pow = 0; pow <= _order; pow++) {
      const PolynomialCoefficient *v =
          static_cast<const PolynomialCoefficient *>(_vertices[pow]);

      const double coefficient = v->estimate();

      if (pow == 0) {
        y_est += coefficient;
      } else {
        y_est += coefficient * std::pow(_x, pow);
      }
    }

    _error(0, 0) = _measurement - y_est;
  }

  virtual void linearizeOplus() override {
    for (int pow = 0; pow <= _order; pow++) {
      if (pow == 0) {
        _jacobianOplus[pow](0, 0) = 1.0;
      }
      _jacobianOplus[pow](0, 0) = std::pow(_x, pow);
    }
  }

  virtual bool read(std::istream &in) { return false; }
  virtual bool write(std::ostream &out) const { return false; }

 private:
  double _x;
  int _order;
};

std::vector<double> fit_poly_g2o(std::vector<double> x_data,
                                 std::vector<double> y_data, int order) {
  if (x_data.size() != y_data.size()) {
    throw std::runtime_error("X and Y must have the same size");
  }
  int N = x_data.size();

  typedef g2o::BlockSolver<g2o::BlockSolverTraits<1, 1>> BlockSolverType;
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
      LinearSolverType;

  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
      std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));

  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(true);

  std::vector<PolynomialCoefficient *> vertices;

  for (int pow = 0; pow <= order; pow++) {
    auto vertex = new PolynomialCoefficient();
    vertex->setEstimate(0);
    vertex->setId(pow);
    optimizer.addVertex(vertex);
    vertices.push_back(vertex);
  }

  for (int i = 0; i < N; i++) {
    const double x = x_data[i];
    const double y = y_data[i];

    auto edge = new PolyFitEdge(x, order);
    edge->setId(i);

    for (int pow = 0; pow <= order; pow++) {
      edge->setVertex(pow, vertices[pow]);
    }

    edge->setMeasurement(y);
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
    optimizer.addEdge(edge);
  }

  optimizer.initializeOptimization();
  optimizer.optimize(10);

  std::cout << "Here" << std::endl;
  std::vector<double> coeffs;

  for (auto &v : vertices) {
    coeffs.push_back(v->estimate());
  }

  return coeffs;
}
}  // namespace foundation