#pragma once
#include <symforce/opt/optimizer.h>

namespace foundation {
inline sym::optimizer_params_t default_optimizer_params() {
  sym::optimizer_params_t params{};
  params.iterations = 50;
  params.verbose = true;
  params.initial_lambda = 1.0;
  params.lambda_update_type = sym::lambda_update_type_t::STATIC;
  params.lambda_up_factor = 10.0;
  params.lambda_down_factor = 1 / 10.0;
  params.lambda_lower_bound = 1.0e-8;
  params.lambda_upper_bound = 1000000.0;
  params.early_exit_min_reduction = 1.0e-6;
  params.use_unit_damping = true;
  params.use_diagonal_damping = false;
  params.keep_max_diagonal_damping = false;
  params.diagonal_damping_min = 1e-6;
  params.enable_bold_updates = false;
  return params;
}
}  // namespace foundation