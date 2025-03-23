#pragma once

#include <Eigen/Core>
#include <foundation/oak_slam/orb.hpp>
#include <foundation/oak_slam/type_defs.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>

namespace foundation {
namespace oak_slam {

std::vector<std::optional<Eigen::Vector3d>> triangulate_triple_match(
    TripleMatch* match,
    OakCameraCalibration* calib,
    double repr_error_threshold
);

std::vector<std::optional<Eigen::Vector3d>> triangulate_normalized_image_points(
    std::vector<std::vector<Eigen::Vector2d>>* normalized_image_points,
    std::vector<Eigen::Matrix4d>* world_to_cameras,
    std::optional<double> max_distance_threshold = std::nullopt
);

}  // namespace oak_slam
}  // namespace foundation