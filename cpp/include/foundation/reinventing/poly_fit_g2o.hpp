#pragma once
#include <vector>

namespace foundation {
std::vector<double> fit_poly_g2o(std::vector<double> x_data,
                                 std::vector<double> y_data, int order);
}
