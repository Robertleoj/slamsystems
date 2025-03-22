#pragma once
#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <foundation/types.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace py = pybind11;

namespace foundation {

template <typename T>
bool is_shape(py::array_t<T, py::array::c_style> arr,
              std::vector<std::optional<int>> shape) {
  if (arr.ndim() != shape.size()) {
    return false;
  }
  for (int i = 0; i < shape.size(); i++) {
    if (shape[i].has_value() && arr.shape(i) != shape[i].value()) {
      return false;
    }
  }
  return true;
}

std::vector<Eigen::Vector3d> array_to_3d_points(double_array arr);

Eigen::Vector3d array_to_3d_point(double_array arr);

std::vector<Eigen::Vector2d> array_to_2d_points(double_array arr);

Eigen::Matrix3d array_to_3mat(double_array arr);

cv::Mat img_numpy_to_mat_uint8(py::array_t<uint8_t, py::array::c_style> input);

py::array_t<uint8_t> mat_to_numpy_uint8(const cv::Mat& mat);

}  // namespace foundation
