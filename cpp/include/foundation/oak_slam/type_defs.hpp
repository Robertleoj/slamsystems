#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <opencv2/opencv.hpp>

#include <foundation/utils/camera.hpp>

namespace py = pybind11;

namespace foundation {
namespace oak_slam {

typedef py::array_t<uint8_t, py::array::c_style> img_array;

struct OakCameraCalibration {
  CameraParams color_intrinsics;
  CameraParams left_intrinsics;
  CameraParams right_intrinsics;
  Eigen::Matrix4d center_to_left;
  Eigen::Matrix4d center_to_right;
};

struct OakFrame {
  cv::Mat center_color;
  cv::Mat left_mono;
  cv::Mat right_mono;
};

struct OakCamLoc {
  enum class Enum { CENTER, LEFT, RIGHT };

  static constexpr Enum CENTER = Enum::CENTER;
  static constexpr Enum LEFT = Enum::LEFT;
  static constexpr Enum RIGHT = Enum::RIGHT;

  static constexpr std::array<Enum, 3> all() { return {CENTER, LEFT, RIGHT}; }
};

}  // namespace oak_slam
}  // namespace foundation