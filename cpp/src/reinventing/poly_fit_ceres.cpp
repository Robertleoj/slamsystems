#include <ceres/ceres.h>

#include <foundation/reinventing/poly_fit_ceres.hpp>
#include <iostream>
#include <memory>
#include <vector>

namespace foundation {
struct PolyFitCost {
 public:
  PolyFitCost(double x_data, double y_data, int order)
      : _x_data(x_data), _y_data(y_data), _order(order) {}

  template <typename T>
  bool operator()(T const* const* coeffs, T* residual) const {
    // leq because order n means n+1 coeffs
    T y_est = T(0);
    for (int i = 0; i <= _order; i++) {
      if (i == 0) {
        y_est += coeffs[0][i];
      } else {
        y_est += coeffs[0][i] * ceres::pow(T(_x_data), i);
      }
    }
    residual[0] = T(_y_data) - y_est;
    return true;
  }

 private:
  const double _x_data, _y_data, _order;
};

std::vector<double> fit_poly_ceres(std::vector<double> x_data,
                                   std::vector<double> y_data, int order) {
  if (x_data.size() != y_data.size()) {
    throw std::runtime_error("X and Y have the same size");
  }
  if (order < 0) {
    throw std::runtime_error("No negative orders bruh");
  }

  int N = x_data.size();

  std::vector<double> coeffs(order + 1, 0.0);

  ceres::Problem problem;

  for (int i = 0; i < N; i++) {
    auto cost_function = new ceres::DynamicAutoDiffCostFunction<PolyFitCost, 4>(
        new PolyFitCost(x_data[i], y_data[i], order));
    cost_function->AddParameterBlock(order + 1);
    cost_function->SetNumResiduals(1);
    problem.AddResidualBlock(cost_function, nullptr, coeffs.data());
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << std::endl;

  std::vector<double> out;
  for (int i = 0; i <= order; i++) {
    out.push_back(coeffs[i]);
  }

  return out;
}
}  // namespace foundation