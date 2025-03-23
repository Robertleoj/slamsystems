#pragma once

#include <foundation/symforce_generated/general_factors/cpp/symforce/general_factors/depth_factor.h>
#include <foundation/symforce_generated/general_factors/cpp/symforce/general_factors/point_reprojection_factor.h>
#include <sym/linear_camera_cal.h>
#include <sym/pose3.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/optimizer.h>

#include <Eigen/Core>
#include <foundation/optim/symforce/conversion.hpp>
#include <foundation/optim/symforce/default_opt_params.hpp>
#include <foundation/types.hpp>
#include <foundation/utils/camera.hpp>
#include <vector>

namespace py = pybind11;

namespace foundation {
namespace depth_slam {

enum Var : char {
  CAMERA_CALIBRATION = 'c',
  CAM_POSE = 'p',
  LANDMARK_LOC = 'l',
  OBSERVATION = 'o',
  DEPTH_OBSERVATION = 'd',
  EPSILON = 'e',
};

struct CameraPose {
  int frame_id;
  Eigen::Matrix4d camera_pose;

  CameraPose(int frame_id, py::EigenDRef<Eigen::Matrix4d> camera_pose)
      : frame_id(frame_id), camera_pose(camera_pose) {}
};

struct Landmark {
  int id;
  Eigen::Vector3d loc;

  Landmark(int id, py::EigenDRef<Eigen::Vector3d> loc) : id(id), loc(loc) {}
};

struct LandmarkProjectionObservation {
  int frame_id;
  int landmark_id;
  Eigen::Vector2d observation;
  std::optional<double> depth;

  LandmarkProjectionObservation(int frame_id, int landmark_id,
                                py::EigenDRef<Eigen::Vector2d> observation,
                                std::optional<double> depth)
      : frame_id(frame_id),
        landmark_id(landmark_id),
        observation(observation),
        depth(depth) {}
};

std::tuple<std::vector<CameraPose>, std::vector<Landmark>> depth_slam_ba(
    CameraParams &cam_calibration, std::vector<CameraPose> &camera_poses,
    std::vector<LandmarkProjectionObservation> &observations,
    std::vector<Landmark> &landmarks) {
  std::vector<sym::Factord> factors;

  for (auto &o : observations) {
    factors.push_back(sym::Factord::Hessian(
        general_factors::PointReprojectionFactor<double>,
        {{Var::CAMERA_CALIBRATION},
         {Var::CAM_POSE, o.frame_id},
         {Var::LANDMARK_LOC, o.landmark_id},
         {Var::OBSERVATION, o.frame_id, o.landmark_id},
         {Var::EPSILON}},
        {{Var::CAM_POSE, o.frame_id}, {Var::LANDMARK_LOC, o.landmark_id}}));

    if (o.depth.has_value()) {
      factors.push_back(sym::Factord::Hessian(
          general_factors::DepthFactor<double>,
          {{Var::CAM_POSE, o.frame_id},
           {Var::LANDMARK_LOC, o.landmark_id},
           {Var::DEPTH_OBSERVATION, o.frame_id, o.landmark_id}},
          {
              {Var::CAM_POSE, o.frame_id},
              {Var::LANDMARK_LOC, o.landmark_id},
          }));
    }
  }

  sym::Valuesd values;

  const double epsilon = 1e-10;
  values.Set({Var::EPSILON}, epsilon);

  values.Set({Var::CAMERA_CALIBRATION}, cam_calibration.to_symforce());

  for (auto &o : observations) {
    values.Set({Var::OBSERVATION, o.frame_id, o.landmark_id},
               sym::Vector2d(o.observation));

    if (o.depth.has_value()) {
      values.Set({Var::DEPTH_OBSERVATION, o.frame_id, o.landmark_id},
                 o.depth.value());
    }
  }

  for (auto &c : camera_poses) {
    values.Set({Var::CAM_POSE, c.frame_id}, pose_from_homo_mat(c.camera_pose));
  }

  for (auto &l : landmarks) {
    values.Set({Var::LANDMARK_LOC, l.id}, sym::Vector3d(l.loc));
  }

  const auto optimizer_params = default_optimizer_params();

  sym::Optimizerd optimizer(optimizer_params, factors, "DepthSlamOptimizer", {},
                            epsilon);

  auto stats = optimizer.Optimize(values);

  std::vector<CameraPose> out_cams;
  std::vector<Landmark> out_landmarks;

  for (auto &c : camera_poses) {
    Eigen::Matrix4d pose = values.At<sym::Pose3d>({Var::CAM_POSE, c.frame_id})
                               .ToHomogenousMatrix();
    CameraPose p = {c.frame_id, pose};
    out_cams.push_back(std::move(p));
  }

  for (auto &l : landmarks) {
    Eigen::Vector3d landmark_loc =
        values.At<sym::Vector3d>({Var::LANDMARK_LOC, l.id});
    Landmark landmark = {l.id, landmark_loc};
    out_landmarks.push_back(std::move(landmark));
  }

  return std::make_tuple(out_cams, out_landmarks);
}

}  // namespace depth_slam
}  // namespace foundation