#include <Remotery.h>
// #include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <format>

#include <Eigen/Core>
#include <foundation/oak_slam/orb.hpp>
#include <foundation/oak_slam/triangulation.hpp>
#include <foundation/oak_slam/type_defs.hpp>
#include <foundation/utils/conversion.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <sophus/se3.hpp>
#include <vector>

namespace foundation {
namespace oak_slam {

std::vector<std::optional<Eigen::Vector3d>> triangulate_triple_match(
    TripleMatch* match,
    OakCameraCalibration* calib,
    double repr_error_threshold
) {
    rmt_ScopedCPUSample(triangulate_triple_match, 0);

    if (match->num_matches() == 0) {
        return std::vector<std::optional<Eigen::Vector3d>>();
    }
    // 1. extract all the points
    std::vector<CameraParams*> intrinsic_calibrations = {
        &calib->left_intrinsics,
        &calib->color_intrinsics,
        &calib->right_intrinsics
    };

    std::vector<cv::KeyPoint>* keypoint_arr[] = {
        &match->match[OakCamLoc::LEFT].keypoints,
        &match->match[OakCamLoc::CENTER].keypoints,
        &match->match[OakCamLoc::RIGHT].keypoints
    };

    std::vector<std::vector<Eigen::Vector2d>> normalized_cam_points;

    for (int i = 0; i < intrinsic_calibrations.size(); i++) {
        auto keypoints = keypoint_arr[i];
        auto calib = intrinsic_calibrations[i];

        std::vector<cv::Point2d> pixels;

        for (auto& kp : *keypoints) {
            pixels.push_back(kp.pt);
        }

        // normalizze points

        std::vector<cv::Point2d> cv_normalized_points;
        cv::undistortPoints(
            pixels,
            cv_normalized_points,
            eigen_to_cv_mat(calib->K),
            eigen_to_cv_mat(calib->dist_coeffs)
        );

        normalized_cam_points.push_back(
            std::move(cv_vecs2_to_eigen(cv_normalized_points))
        );
    }

    std::vector<Eigen::Matrix4d> cam_positions = {
        calib->center_to_left,
        Eigen::Matrix4d::Identity(),
        calib->center_to_right
    };

    // 3. triangulate rays, keeping only points with no outliers
    auto triangulated = triangulate_normalized_image_points(
        &normalized_cam_points, &cam_positions, 50.0
    );

    // 4. filter out the ones with bad reprojection error
    for (int point_idx = 0; point_idx < triangulated.size(); point_idx++) {
        if (!triangulated[point_idx].has_value()) {
            continue;
        }

        bool bad = false;
        // get the reprojection error for every camera
        for (int camera_idx = 0; camera_idx < intrinsic_calibrations.size();
             camera_idx++) {
            auto& measured = keypoint_arr[camera_idx]->at(point_idx).pt;
            auto& point_in_world = triangulated[point_idx].value();
            auto cam_se3 = Sophus::SE3d(cam_positions[camera_idx]);

            auto rvec = cam_se3.inverse().so3().log();

            auto tvec = cam_se3.inverse().translation();

            cv::Mat cv_out_mat;

            cv::projectPoints(
                cv::Mat(eigen_vec3_to_cv(point_in_world)),
                eigen_to_cv_mat(rvec),
                eigen_to_cv_mat(tvec),
                eigen_to_cv_mat(intrinsic_calibrations[camera_idx]->K),
                eigen_to_cv_mat(intrinsic_calibrations[camera_idx]->dist_coeffs
                ),
                cv_out_mat
            );

            cv::Point2d cv_out_point(
                cv_out_mat.at<double>(0, 0), cv_out_mat.at<double>(0, 1)
            );

            double repr_error = cv::norm(cv::Point2d(measured) - cv_out_point);
            if (repr_error > repr_error_threshold) {
                spdlog::debug(std::format(
                    "bad reprojection error: {} camera {}",
                    repr_error,
                    camera_idx
                ));
                bad = true;
                break;
            }
        }

        if (bad) {
            triangulated[point_idx] = std::nullopt;
        }
    }

    return triangulated;
}

std::vector<std::optional<Eigen::Vector3d>> triangulate_normalized_image_points(
    std::vector<std::vector<Eigen::Vector2d>>* normalized_image_points,
    std::vector<Eigen::Matrix4d>* world_to_cameras,
    std::optional<double> max_distance_threshold
) {
    rmt_ScopedCPUSample(triangulate_normalized_image_points, 0);
    int num_cameras = world_to_cameras->size();
    int num_points = normalized_image_points->at(0).size();

    std::vector<Sophus::SE3d> se3s;
    for (auto& world_to_camera : *world_to_cameras) {
        se3s.push_back(Sophus::SE3d(world_to_camera));
    }

    std::vector<std::optional<Eigen::Vector3d>> triangulated_points(num_points);

    for (int point_idx = 0; point_idx < num_points; point_idx++) {
        Eigen::Matrix3d AtA = Eigen::Matrix3d::Zero();
        Eigen::Vector3d Atb = Eigen::Vector3d::Zero();

        std::vector<Eigen::Vector3d> ray_origins;
        std::vector<Eigen::Vector3d> ray_directions;

        for (int cam_idx = 0; cam_idx < num_cameras; cam_idx++) {
            Eigen::Vector2d xy =
                normalized_image_points->at(cam_idx)[point_idx];
            Eigen::Vector3d d =
                Eigen::Vector3d(xy.x(), xy.y(), 1.0).normalized();

            Eigen::Vector3d p1 = se3s[cam_idx].translation();
            Eigen::Vector3d p2 = p1 + se3s[cam_idx].so3() * d;

            Eigen::Vector3d ray_dir = (p2 - p1).normalized();
            Eigen::Matrix3d A =
                Eigen::Matrix3d::Identity() - ray_dir * ray_dir.transpose();
            Eigen::Vector3d b = A * p1;

            AtA += A;
            Atb += b;

            ray_origins.push_back(p1);
            ray_directions.push_back(ray_dir);
        }

        if (AtA.determinant() < 1e-6) {  // Ensure solvability
            triangulated_points[point_idx] = std::nullopt;  // Degenerate case
            continue;
        }

        Eigen::Vector3d X = AtA.ldlt().solve(Atb);

        if (!max_distance_threshold.has_value()) {
            triangulated_points[point_idx] = X;
            continue;
        }

        // Compute max distance from the triangulated point to each ray
        double max_distance = 0.0;
        for (int cam_idx = 0; cam_idx < num_cameras; cam_idx++) {
            Eigen::Vector3d p1 = ray_origins[cam_idx];
            Eigen::Vector3d dir = ray_directions[cam_idx];

            // Compute perpendicular distance from X to ray
            Eigen::Vector3d diff = X - p1;
            Eigen::Vector3d proj =
                dir * (diff.dot(dir));           // Projection onto the ray
            double dist = (diff - proj).norm();  // Perpendicular distance

            max_distance = std::max(max_distance, dist);
        }

        // Check if max distance is within the threshold
        if (max_distance < max_distance_threshold.value()) {
            triangulated_points[point_idx] = X;
        } else {
            triangulated_points[point_idx] = std::nullopt;
        }
    }

    return triangulated_points;
}

}  // namespace oak_slam
}  // namespace foundation